# %%

import utils
from architectures.bloom_560m import Bloom560m
from config.bloom560m_easyEN2SP import get_trainer, MLFlowExperimentManager
from data_processing.easy_ds_EN_to_SP import EasyEnToSpDM

utils.ini_setting()

def run_training():
    model = Bloom560m("D:/models", learning_rate=0.001)
    data_module = EasyEnToSpDM(tokenizer=model.tokenizer, batch_size=1)
    trainer = get_trainer()

    with MLFlowExperimentManager():
        trainer.fit(model=model, datamodule=data_module)
        trainer.test(model=model, datamodule=data_module)
        predictions = trainer.predict(model=model, datamodule=data_module, return_predictions=True)

    display_predictions(predictions)

def display_predictions(predictions):
    for prediction in predictions:
        print(prediction)

if __name__ == "__main__":
    run_training()
    pass


