import pandas as pd
import joblib
import os
import logging
from datetime import datetime
import hopsworks
from utils.config import Config

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Class for managing the model registry in Hopsworks."""

    def __init__(self, project_name=None, api_key=None):
        """Initialize the model registry."""
        self.project_name = project_name or Config.HOPSWORKS_PROJECT_NAME
        self.api_key = api_key or Config.HOPSWORKS_API_KEY
        self.model_registry_name = Config.MODEL_REGISTRY_NAME
        self.project = None
        self.mr = None

        self._connect()

    def _connect(self):
        """Connect to Hopsworks project and model registry."""
        try:
            self.project = hopsworks.login(
                project=self.project_name,
                api_key_value=self.api_key
            )
            self.mr = self.project.get_model_registry()
            logger.info(f"Connected to Hopsworks project: {self.project_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            self.project = None
            self.mr = None

    def save_model(self, model, name, metrics=None, tags=None, description=None, version=None):
        """Save a model to the registry."""
        if self.mr is None:
            logger.error("Not connected to Hopsworks model registry")
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            local_model_path = f"/tmp/{name}_{timestamp}.joblib"

            joblib.dump(model, local_model_path)

            model_metrics = metrics or {}
            model_tags = tags or {}
            model_tags['timestamp'] = timestamp

            if version:
                model_dir = self.mr.get_or_create_model(
                    name=name,
                    version=version,
                    metrics=model_metrics,
                    description=description or f"Model {name} v{version}"
                )
            else:
                model_dir = self.mr.python.create_model(
                    name=name,
                    metrics=model_metrics,
                    description=description or f"Model {name}"
                )

            model_dir.save(local_model_path)

            # Safe tagging support
            for tag_key, tag_value in model_tags.items():
                try:
                    if hasattr(model_dir, "add_tag"):
                        model_dir.add_tag(name=tag_key, value=str(tag_value))
                    else:
                        logger.debug(f"Skipping tag {tag_key} — add_tag not supported.")
                except Exception as tag_err:
                    logger.warning(f"Could not add tag {tag_key}: {tag_err}")

            logger.info(f"Saved model {name} to registry")

            if os.path.exists(local_model_path):
                os.remove(local_model_path)

            return model_dir

        except Exception as e:
            logger.error(f"Failed to save model to registry: {e}")
            return None

    def load_model(self, name, version=None):
        """Load a model from the registry."""
        if self.mr is None:
            logger.error("Not connected to Hopsworks model registry")
            return None

        try:
            if version:
                model_dir = self.mr.get_model(name=name, version=version)
            else:
                model_dir = self.mr.get_best_model(name=name)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            local_model_path = f"/tmp/{name}_{timestamp}.joblib"

            model_dir.download()
            downloaded_model_path = model_dir.get_downloaded_model_path()

            model = joblib.load(downloaded_model_path)

            logger.info(f"Loaded model {name} from registry")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}")
            return None

    def get_model_versions(self, name):
        """Get all versions of a model."""
        if self.mr is None:
            logger.error("Not connected to Hopsworks model registry")
            return []

        try:
            models = self.mr.get_models(name=name)
            return models
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []

    def get_best_model(self, name, metric='rmse', ascending=True):
        """Get the best model version based on a metric."""
        if self.mr is None:
            logger.error("Not connected to Hopsworks model registry")
            return None

        try:
            models = self.mr.get_models(name=name)

            if not models:
                logger.warning(f"No models found with name: {name}")
                return None

            def get_metric(model):
                try:
                    return float(model.get_metrics().get(metric, float('inf')))
                except:
                    return float('inf') if ascending else float('-inf')

            sorted_models = sorted(models, key=get_metric, reverse=not ascending)

            if sorted_models:
                return sorted_models[0]
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            return None
