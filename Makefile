data:
	python src/data_download.py

clean:
	python src/data_cleaning.py

features:
	python src/feature_engineering.py

train:
	python src/model_training.py

validate:
	python src/model_validation.py

explain:
	python src/model_explainability.py

treatment:
	python src/treatment_effect_analysis.py

report:
	python src/generate_report.py


all:
	make data
	make clean
	make features
	make train
	make validate
	make explain
	make treatment
	make report