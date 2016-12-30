from gensim.models import Word2Vec
import sys
model = Word2Vec.load(sys.argv[1])
while (True):
	try:
		x = raw_input("Enter a word : ")
		if x in model:
			print model[x]
			#print "Most similar words with '%s' is : %s" %(x, model.most_similar(positive=[x]))
		else:
			print "Not found in model!"
	except EOFError:
		print "Bye~"
		break
