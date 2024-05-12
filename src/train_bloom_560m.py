# %%

import utils
from architectures.bloom_560m import Bloom560m
from config.bloom560m_easyEN2SP import get_trainer,MLFlowExperimentManager
from data_processing.easy_ds_EN_to_SP import EasyEnToSpDM



utils.ini_setting()

def training_loop():

    model = Bloom560m("D:/models", 0.001)
    easy_en_to_sp_dm = EasyEnToSpDM(model.tokenizer, 1)
    trainer = get_trainer()
    with MLFlowExperimentManager():
        trainer.fit(model=model, datamodule=easy_en_to_sp_dm)
        trainer.test(model=model, datamodule=easy_en_to_sp_dm)
        predictions = trainer.predict(
            model=model,
            datamodule=easy_en_to_sp_dm,
            return_predictions=True,
        )
    for pred in predictions:
        print(pred)


if __name__ == "__main__":
    training_loop()

