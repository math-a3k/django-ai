# -*- coding: utf-8 -*-

def is_float(value):
	try:
		float(value)
		return(True)
	except Exception as e:
		return(False)
