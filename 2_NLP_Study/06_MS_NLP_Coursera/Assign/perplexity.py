import sys

def main():
    if len(sys.argv) < 2:
        print "Usage: python perplexity.py <file of scores> <file of sentences that were scored>"
        exit(1)
    
    infile = open(sys.argv[1], "r")
    scores = infile.readlines()
    infile.close()
    infile = open(sys.argv[2], 'r')
    sentences = infile.readlines()
    infile.close()
    
    M = 0

    for sentence in sentences:
        words = sentence.split()
        M += len(words) + 1

    perplexity = 0
    for score in scores:
       perplexity += float(score.split()[0])  # assume log probability

    perplexity /= M
    perplexity = 2 ** (-1 * perplexity)

    print "The perplexity is", perplexity    
	 
if __name__ == "__main__": main()
