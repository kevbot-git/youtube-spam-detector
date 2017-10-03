import sys, glob, csv, codecs

def main():
	model_filenames, deploy_filenames = sysargs()
	model = create_model(model_filenames)

def create_model(filenames):
	data = extract_data(filenames)

# Returns an array of tuples: [('text here 123 testing', true)]
# The first tuple entry is a comment's text, the second classifies it as SPAM or HAM
# Data file structure:	COMMENT_ID,	AUTHOR,		DATE,		CONTENT,	CLASS
# 						0			1			2			3			4
def extract_data(filenames):
	for i in range(0, len(filenames)):
		print('Loading \'' + filenames[i] + '\'')
		with codecs.open(filenames[i], 'r', encoding='utf-8') as f: # Was having trouble with encoding, this solves issues
			file_entries = csv.reader(f)
			for line in file_entries:
				print(line)
	return 0

# Returns a tuple: (model_filename, deploy_filename)
def sysargs():
	if (len(sys.argv) == 3):
		model_f = glob.glob(sys.argv[1])
		deploy_f = glob.glob(sys.argv[2])
		if (model_f != [] and deploy_f != []):
			return (model_f, deploy_f)
		else:
			print('Error: search strings must match at least one file each.')
			exit(1)
	else:
		print('Usage: python model_on_and_deploy_to.py <training_set> <deploy_set>')
		print('Note: search patterns such as ./*.csv can be used to supply multiple files for each set.')
		exit(1)

if (__name__ == '__main__'):
    main()
