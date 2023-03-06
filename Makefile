install_package:
	@pip uninstall -y ddd || :
	@pip install -e .

test_bucket_connexion:
	@python ddd/interface/main.py
