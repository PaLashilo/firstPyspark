from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.functions import col, when
from pyspark.sql import DataFrame

class CustomEncoder(Transformer, HasInputCol, HasOutputCol):
    def __init__(self, encoding_dict=None, inputCol=None, outputCol=None):
        super(CustomEncoder, self).__init__()
        self.encoding_dict = encoding_dict
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _transform(self, df: DataFrame) -> DataFrame:
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()
        
        encoded_col = col(inputCol)
        for category, encoded_value in self.encoding_dict.items():
            encoded_col = when(col(inputCol) == category, encoded_value).otherwise(encoded_col)
        
        return df.withColumn(outputCol, encoded_col)
