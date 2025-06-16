# DRML-Ensemble

1、 File Description:

-CommonHHJ: Toolkit for reading and writing CSV files

-FallGraph: The FGraph dataset

-CallGraph: The CGraph dataset

Each dataset includes the following files:

DTI.csv drug target (protein) isomerization relationship

DGI. csv: Disease Gene (Protein) Isomerization Relationship

DrugDisI.csv drug disease heterogeneity relationship

DrugFeature.csv Drug Node Initial Features

DiseaseFeature.csv Disease Node Initial Features

Ensure that all the above files exist during runtime.

-Models: Model code

-Src: Dataset read code

-Main. py model training and testing code

-ReprositionMain.py Drug Relocation Prediction Code



2、 Code Running

Running environment: pytorch, DGL cu111, numpy, pandas

Parameter configuration settings can be modified directly by setting default values in the code, or by using the "Python main. py -- datasetRoot * * * *" command.

Mainly set the datasetRoot parameter and adjust the dataset; Set the type parameter to save the path; Top K adjustment preserves the k value of node feature top k; Alphe adjusts the feature retention coefficient in the model.
