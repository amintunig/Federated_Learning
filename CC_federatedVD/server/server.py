import flwr as fl
import torch
from typing import List, Tuple, Optional, Dict, Any, Union
from model.gan import get_gan_models
from config import DEVICE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GanServer(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,  # Use all available clients for training
            fraction_evaluate=0.5,  # Use half of clients for evaluation
            min_fit_clients=2,  # Minimum clients needed for training
            min_evaluate_clients=1,  # Minimum clients needed for evaluation
            min_available_clients=2,  # Minimum total clients needed
        )
        self.G, self.D = get_gan_models()
        self.G.to(DEVICE)
        self.D.to(DEVICE)
        logger.info("GAN models initialized on device: %s", DEVICE)

    def initialize_parameters(self, client_manager) -> Optional[fl.common.Parameters]:
        """Initialize global model parameters."""
        logger.info("Initializing global parameters")
        
        # Combine G and D parameters
        params = [val.cpu().numpy() for val in self.G.state_dict().values()]
        params += [val.cpu().numpy() for val in self.D.state_dict().values()]
        
        return fl.common.ndarrays_to_parameters(params)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results using weighted average."""
        
        if not results:
            logger.warning("No fit results received in round %d", server_round)
            return None, {}
        
        logger.info("Aggregating fit results from %d clients in round %d", len(results), server_round)
        
        # Call parent's aggregate_fit method
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            try:
                # Convert Parameters to list of ndarrays
                aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Split the list into G and D parts
                g_keys = list(self.G.state_dict().keys())
                d_keys = list(self.D.state_dict().keys())
                g_len = len(g_keys)

                if len(aggregated_ndarrays) != g_len + len(d_keys):
                    logger.error("Parameter count mismatch: expected %d, got %d", 
                               g_len + len(d_keys), len(aggregated_ndarrays))
                    return None, {}

                g_params = aggregated_ndarrays[:g_len]
                d_params = aggregated_ndarrays[g_len:]

                # Create state dictionaries
                g_state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in zip(g_keys, g_params)}
                d_state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in zip(d_keys, d_params)}

                # Load state dictionaries
                self.G.load_state_dict(g_state_dict, strict=True)
                self.D.load_state_dict(d_state_dict, strict=True)
                
                logger.info("Successfully updated global models in round %d", server_round)
                
            except Exception as e:
                logger.error("Error updating global models: %s", str(e))
                return None, {}

        # Aggregate metrics
        total_examples = 0
        total_g_loss = 0.0
        total_d_loss = 0.0
        
        for _, fit_res in results:
            total_examples += fit_res.num_examples
            if "generator_loss" in fit_res.metrics:
                total_g_loss += fit_res.metrics["generator_loss"] * fit_res.num_examples
            if "discriminator_loss" in fit_res.metrics:
                total_d_loss += fit_res.metrics["discriminator_loss"] * fit_res.num_examples

        if total_examples > 0:
            avg_g_loss = total_g_loss / total_examples
            avg_d_loss = total_d_loss / total_examples
            aggregated_metrics.update({
                "generator_loss": avg_g_loss,
                "discriminator_loss": avg_d_loss,
            })

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results."""
        
        if not results:
            logger.warning("No evaluation results received in round %d", server_round)
            return None, {}
        
        logger.info("Aggregating evaluation results from %d clients in round %d", len(results), server_round)
        
        # Aggregate metrics
        total_examples = 0
        total_accuracy = 0.0
        
        for _, eval_res in results:
            total_examples += eval_res.num_examples
            if "accuracy" in eval_res.metrics:
                total_accuracy += eval_res.metrics["accuracy"] * eval_res.num_examples

        if total_examples > 0:
            avg_accuracy = total_accuracy / total_examples
            return None, {"accuracy": avg_accuracy}
        
        return None, {}

def main():
    """Start the Flower server."""
    strategy = GanServer()

    try:
        fl.server.start_server(
            server_address="0.0.0.0:8084",
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=strategy,
        )
    except Exception as e:
        logger.error("Error starting server: %s", str(e))
        raise

if __name__ == "__main__":
    main()