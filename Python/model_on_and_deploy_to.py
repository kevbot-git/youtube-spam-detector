import sys

def main():
	model_filename, deploy_filename = sysargs()

def sysargs():
	if (len(sys.argv) == 3):
		return (str(sys.argv[1]), str(sys.argv[2]))
	else:
		print('Usage: python model_on_and_deploy_to.py <training_set> <deploy_set>')

if __name__ == '__main__':
    main()
