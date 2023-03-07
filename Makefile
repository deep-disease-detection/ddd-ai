install_package:
	@pip uninstall -y ddd || :
	@pip install -e .
