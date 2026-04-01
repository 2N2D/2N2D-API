import logging
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Models.UniversalArchitectureModel import UniversalArchitectureModel
import random
import copy
from Optimize.GraphMutator import FXMutator

try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False
    logging.warning("DEAP not available. Install with: pip install deap")

AVAILABLE_MUTATIONS = [
    'scale_width_0.75',
    'scale_width_1.25',
    'scale_width_1.5',
    'scale_width_2.0',
    'add_layer',
    'remove_layer',
    'add_skip_connection'
]

def optimize_with_genetic_deap(X_train, y_train, input_size, output_size, base_model_info, population_size=20, generations=10, status_callback=None, device=None, is_pytorch=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if not HAS_DEAP:
        raise ImportError("DEAP library is required for genetic algorithm optimization. Please install it with: pip install deap")
    base_layers = base_model_info.get('num_layers', 1)
    base_neurons = base_model_info.get('hidden_size', 64)
    model_type = base_model_info.get('model_type', 'feedforward')
    
    logging.info(f"DEAP GA starting with base: {base_layers} layers, {base_neurons} neurons")
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    def create_individual():
        # A list of 1 to 4 random structural mutation commands
        num_mutations = random.randint(1, 4)
        return creator.Individual(random.choices(AVAILABLE_MUTATIONS, k=num_mutations))
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate_individual(individual):
        try:
            mutations = list(individual)

            if is_pytorch and '_native_fx_model' in base_model_info:
                base_fx = copy.deepcopy(base_model_info['_native_fx_model'])
                mutator = FXMutator()
                # Apply the explicit list of structural graph mutations
                model = mutator.apply_mutations(base_fx, mutations)
                model = model.to(device)
            else:
                # Legacy support fallback mapping: simulate width operations
                enhancement_factor = 1.0
                for mut in mutations:
                    if mut.startswith('scale_width_'):
                        enhancement_factor *= float(mut.split('_')[-1])
                
                enhanced_arch_info = base_model_info.copy()
                enhanced_arch_info['enhancement_pattern'] = random.randint(0, 3)
                model = UniversalArchitectureModel(
                    architecture_info=enhanced_arch_info,
                    enhancement_factor=enhancement_factor,
                    actual_input_size=input_size,
                    actual_output_size=output_size
                )
                model = model.to(device)
                if hasattr(model, 'input_adapter'):
                    adapter_device = next(model.input_adapter.parameters()).device
                    main_device = next(model.parameters()).device
                    if adapter_device != main_device:
                        model.input_adapter = model.input_adapter.to(main_device)
            
            val_size = int(0.2 * len(X_train))
            X_train_eval = X_train[val_size:]
            X_val_eval = X_train[:val_size]
            y_train_eval = y_train[val_size:]
            y_val_eval = y_train[:val_size]
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            model.train()
            
            for epoch in range(12):
                total_loss = 0
                batch_count = 0
                for i in range(0, len(X_train_eval), 32):
                    end_idx = min(i + 32, len(X_train_eval))
                    X_batch = torch.tensor(X_train_eval[i:end_idx], dtype=torch.float32, device=device)
                    y_batch = torch.tensor(y_train_eval[i:end_idx], dtype=torch.float32, device=device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    
                    if outputs.shape[1] != y_batch.shape[1]:
                        logging.warning(f"Shape mismatch detected: model outputs {outputs.shape[1]}, target needs {y_batch.shape[1]}")
                        if outputs.shape[1] > y_batch.shape[1]:
                            outputs = outputs[:, :y_batch.shape[1]]
                        else:
                            padding_size = y_batch.shape[1] - outputs.shape[1]
                            padding = torch.zeros(outputs.shape[0], padding_size, device=device)
                            outputs = torch.cat([outputs, padding], dim=1)
                    
                    loss = nn.MSELoss()(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_loss += loss.item()
                    batch_count += 1
            model.eval()
            with torch.no_grad():
                all_outputs = []
                all_targets = []
                for i in range(0, len(X_val_eval), 32):
                    end_idx = min(i + 32, len(X_val_eval))
                    X_batch = torch.tensor(X_val_eval[i:end_idx], dtype=torch.float32, device=device)
                    y_batch = torch.tensor(y_val_eval[i:end_idx], dtype=torch.float32, device=device)
                    
                    outputs = model(X_batch)
                    if outputs.shape[1] != y_batch.shape[1]:
                        if outputs.shape[1] > y_batch.shape[1]:
                            outputs = outputs[:, :y_batch.shape[1]]
                        else:
                            padding_size = y_batch.shape[1] - outputs.shape[1]
                            padding = torch.zeros(outputs.shape[0], padding_size, device=device)
                            outputs = torch.cat([outputs, padding], dim=1)
                    
                    all_outputs.append(outputs.cpu().numpy())
                    all_targets.append(y_batch.cpu().numpy())
                
                all_outputs = np.vstack(all_outputs)
                all_targets = np.vstack(all_targets)
                
                from sklearn.metrics import r2_score
                r2 = r2_score(all_targets, all_outputs)
                # To keep DEAP selection mostly positive, clip strongly negative R2
                fitness = max(0.0001, r2)
                
            return (fitness,)
        except Exception as e:
            logging.warning(f"Error evaluating DEAP individual: {e}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")
            return (0.0001,)  
    toolbox.register("evaluate", evaluate_individual)
    
    def custom_mate(ind1, ind2):
        '''Crossover by randomly mixing the list of mutation actions'''
        size1, size2 = len(ind1), len(ind2)
        if size1 > 0 and size2 > 0:
            cxpoint1 = random.randint(0, size1 - 1)
            cxpoint2 = random.randint(0, size2 - 1)
            ind1[cxpoint1:], ind2[cxpoint2:] = ind2[cxpoint2:], ind1[cxpoint1:]
        return ind1, ind2

    def custom_mutate(individual):
        """Randomly alters the list of mutation instructions"""
        if random.random() < 0.5 and len(individual) > 0: 
            # Replace an existing mutation
            mutate_idx = random.randint(0, len(individual) - 1)
            individual[mutate_idx] = random.choice(AVAILABLE_MUTATIONS)
        elif random.random() < 0.25 and len(individual) < 8:
            # Append a new mutation
            individual.append(random.choice(AVAILABLE_MUTATIONS))
        elif random.random() < 0.25 and len(individual) > 1:
            # Remove a random mutation
            individual.pop(random.randint(0, len(individual) - 1))
        return (individual,)
    
    toolbox.register("mate", custom_mate)
    toolbox.register("mutate", custom_mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    population = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    best_fitness_history = []
    best_individual = None
    best_fitness = 0
    
    for gen in range(generations):
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        current_best = max(population, key=lambda x: x.fitness.values[0])
        if current_best.fitness.values[0] > best_fitness:
            best_fitness = current_best.fitness.values[0]
            best_individual = current_best[:]
        
        best_fitness_history.append(best_fitness)
        
        
        if status_callback:
            progress = 20 + int((gen / generations) * 50)  
            status_callback({
                "status": f"Genetic Generation {gen+1}/{generations}: Evolving architectures...",
                "progress": progress
            })
        
        logging.info(f"DEAP Gen {gen+1}: Best Fitness = {best_fitness:.4f} "
                    f"(Mutations={best_individual})")
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.3:  
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        population[:] = offspring
    
    mutations = list(best_individual)
    if is_pytorch and '_native_fx_model' in base_model_info:
        base_fx = copy.deepcopy(base_model_info['_native_fx_model'])
        mutator = FXMutator()
        best_model = mutator.apply_mutations(base_fx, mutations)
        enhancement_factor = 1.0 # placeholder
        pattern = 1
    else:
        enhancement_factor = 1.0
        for mut in mutations:
            if mut.startswith('scale_width_'):
                enhancement_factor *= float(mut.split('_')[-1])
        pattern = 1
        enhanced_arch_info = base_model_info.copy()
        enhanced_arch_info['enhancement_pattern'] = pattern
        best_model = UniversalArchitectureModel(
            architecture_info=enhanced_arch_info,
            enhancement_factor=enhancement_factor,
            actual_input_size=input_size,
            actual_output_size=output_size
        )
    
    best_model = best_model.to(device)
    
    
    if hasattr(best_model, 'input_adapter'):
        adapter_device = next(best_model.input_adapter.parameters()).device
        main_device = next(best_model.parameters()).device
        logging.info(f"Final model device check - main model: {main_device}, input_adapter: {adapter_device}")
        if adapter_device != main_device:
            logging.warning(f"Final model device mismatch! Moving input_adapter from {adapter_device} to {main_device}")
            best_model.input_adapter = best_model.input_adapter.to(main_device)
    
    
    actual_layers = getattr(best_model, '_enhanced_layers', base_model_info.get('num_layers', 1))
    actual_neurons = getattr(best_model, '_enhanced_width', base_model_info.get('max_width', 64))
    
    
    best_model._genetic_architecture = {
        'layers': actual_layers,
        'neurons': actual_neurons,
        'enhancement_factor': enhancement_factor,
        'enhancement_pattern': pattern
    }
    
    logging.info(f"DEAP final best: enhancement factor {enhancement_factor:.2f}, pattern {pattern}")
    logging.info(f"Enhanced architecture: {actual_layers} layers, {actual_neurons} max neurons")
    logging.info(f"Universal architecture created for any ONNX model type")
    
    return best_model, best_fitness_history