# device/src/virtual_space.py
class VirtualSpace:
    def __init__(self, veector, model_manager=None):
        self.veector = veector
        self.model_manager = model_manager

    def perform_inference(self, model_name, input_data):
        if self.model_manager:
            return self.model_manager.perform_inference(model_name, input_data)
        else:
            print("ModelManager not initialized.")
            return None
