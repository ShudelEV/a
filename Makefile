.PHONY: runserver

runserver:
	gunicorn --reload object_on_image:app
