update_env:
	conda env export -f builds/environment.yml

build_env:
	conda env create -f builds/environment.yml
	
reverse_zip:
	python get_reverse_zip.py

impute_zip: reverse_zip
	python impute_zip.py
