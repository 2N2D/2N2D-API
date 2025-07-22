import logging
import neat
import torch.nn as nn
from Models.NEATPytorchModel import NEATPytorchModel
import torch.optim as optim
import torch

def optimize_with_neat(X_train, y_train, input_size, output_size, config_path, base_model_info, generations=5, status_callback=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    current_generation = [0]  
    def eval_genomes(genomes, config):
        current_generation[0] += 1
        
        
        if status_callback:
            progress = 20 + int((current_generation[0] / generations) * 50)  
            status_callback({
                "status": f"NEAT Generation {current_generation[0]}/{generations}: Evolving neural networks...",
                "progress": progress
            })
        
        for genome_id, genome in genomes:
            try:
                net = NEATPytorchModel(genome, config, input_size, output_size, base_model_info)
                net = net.to(device)  
                optimizer = optim.Adam(net.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                
                val_size = int(0.2 * len(X_train))
                X_train_eval = X_train[val_size:]
                X_val_eval = X_train[:val_size]
                y_train_eval = y_train[val_size:]
                y_val_eval = y_train[:val_size]
                
                net.train()
                for epoch in range(15):  
                    epoch_loss = 0
                    batch_count = 0
                    for i in range(0, len(X_train_eval), 32):
                        end_idx = min(i + 32, len(X_train_eval))
                        X_batch = torch.tensor(X_train_eval[i:end_idx], dtype=torch.float32, device=device)
                        y_batch = torch.tensor(y_train_eval[i:end_idx], dtype=torch.float32, device=device)
                        
                        optimizer.zero_grad()
                        outputs = net(X_batch)
                        if outputs.shape[1] != y_batch.shape[1]:
                            logging.warning(f"Shape mismatch detected: model outputs {outputs.shape[1]}, target needs {y_batch.shape[1]}")
                            if outputs.shape[1] > y_batch.shape[1]:
                                outputs = outputs[:, :y_batch.shape[1]]
                            else:
                                padding_size = y_batch.shape[1] - outputs.shape[1]
                                padding = torch.zeros(outputs.shape[0], padding_size, device=device)
                                outputs = torch.cat([outputs, padding], dim=1)
                        
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        epoch_loss += loss.item()
                        batch_count += 1
                
                net.eval()
                total_loss = 0
                eval_batches = 0
                with torch.no_grad():
                    for i in range(0, len(X_val_eval), 32):
                        end_idx = min(i + 32, len(X_val_eval))
                        X_batch = torch.tensor(X_val_eval[i:end_idx], dtype=torch.float32, device=device)
                        y_batch = torch.tensor(y_val_eval[i:end_idx], dtype=torch.float32, device=device)
                        
                        outputs = net(X_batch)
                        if outputs.shape[1] != y_batch.shape[1]:
                            if outputs.shape[1] > y_batch.shape[1]:
                                outputs = outputs[:, :y_batch.shape[1]]
                            else:
                                padding_size = y_batch.shape[1] - outputs.shape[1]
                                padding = torch.zeros(outputs.shape[0], padding_size, device=device)
                                outputs = torch.cat([outputs, padding], dim=1)
                        
                        loss = criterion(outputs, y_batch)
                        total_loss += loss.item()
                        eval_batches += 1
                avg_loss = total_loss / max(eval_batches, 1)
                genome.fitness = 1.0 / (avg_loss + 1e-5)
                if genome_id <= 5:
                    logging.info(f"NEAT Genome {genome_id}: Val_Loss={avg_loss:.4f}, Fitness={genome.fitness:.4f}")
                
            except Exception as e:
                genome.fitness = 1e-6
                logging.warning(f"Error evaluating genome {genome_id}: {e}")
    winner = p.run(eval_genomes, generations)
    winner_model = NEATPytorchModel(winner, config, input_size, output_size, base_model_info)
    
    
    winner_model = winner_model.to(device)
    
    
    if hasattr(winner_model, 'enhanced_model') and hasattr(winner_model.enhanced_model, 'input_adapter'):
        adapter_device = next(winner_model.enhanced_model.input_adapter.parameters()).device
        main_device = next(winner_model.parameters()).device
        logging.info(f"NEAT final model device check - main model: {main_device}, input_adapter: {adapter_device}")
        if adapter_device != main_device:
            logging.warning(f"NEAT final model device mismatch! Moving input_adapter from {adapter_device} to {main_device}")
            winner_model.enhanced_model.input_adapter = winner_model.enhanced_model.input_adapter.to(main_device)
    
    
    enabled_connections = [c for c in winner.connections.values() if c.enabled]
    total_nodes = len(winner.nodes)
    
    
    hidden_nodes = total_nodes - input_size - output_size
    
    
    enhancement_factor = getattr(winner_model, '_enhancement_factor', 1.0)
    if hasattr(winner_model, 'enhanced_model'):
        actual_layers = getattr(winner_model.enhanced_model, '_enhanced_layers', base_model_info.get('num_layers', 1))
        actual_neurons = getattr(winner_model.enhanced_model, '_enhanced_width', base_model_info.get('max_width', 64))
    else:
        
        actual_layers = max(1, int(base_model_info.get('num_layers', 1) * enhancement_factor))
        actual_neurons = max(64, int(base_model_info.get('max_width', 64) * enhancement_factor))
    
    logging.info(f"NEAT optimization complete. Final architecture:")
    logging.info(f"  NEAT genome: {total_nodes} nodes, {len(enabled_connections)} connections, {hidden_nodes} hidden nodes")
    logging.info(f"  Enhanced PyTorch model: {actual_layers} layers, {actual_neurons} max neurons")
    logging.info("Visualization disabled to reduce dependencies.")
    
    
    winner_model._neat_architecture = {
        'layers': actual_layers,
        'neurons': actual_neurons,
        'neat_nodes': total_nodes,
        'neat_connections': len(enabled_connections),
        'neat_hidden_nodes': hidden_nodes,
        'enhancement_factor': enhancement_factor
    }
    
    return winner_model