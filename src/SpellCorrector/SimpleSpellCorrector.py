import enchant
from difflib import SequenceMatcher
import json
from pdb import set_trace
import re
import argparse

class SpellCorrector:
	def __init__(self, lexicon="lexicon.txt", misclassify="misclassify.json"):
		"""
		This SimpleSpellCorrector takes as input the predicted character outputs from the CNN concatenated together
		to form a single word, and tries to spell correct the word. It takes advantage of common misclassifications
		made by the CNN such as classifying "O" as "0", "I" as "1", "S" as "5", etc. 

		It also checks against a lexicon of user-supplied words so that we reduce the likelihood of mis-correcting valid
		outputs such as chinese names, NRICs, dates, etc.

		Arguments:
			lexicon -  a txt document containing domain-specific words that are not found in US english dictionary, 
					   but are considered valid. Words in the lexicon will be added to the spell checker dictionary.
			misclassify - a json document containing the common misclassifications made by the CNN. The default json file
						  was obtained after analysing the beta_ConfusionMatrix.csv file found under the analysis folder

		Returns:
			None
		"""

		self.checker = enchant.DictWithPWL("en_US", lexicon)
		self.misclassify = json.load(open(misclassify))

	def correct(self, text):
		"""
		Applies PyEnchant's spell checking algorithm: https://faculty.math.illinois.edu/~gfrancis/illimath/windows/aszgard_mini/movpy-2.0.0-py2.4.4/manuals/PyEnchant/PyEnchant%20Tutorial.htm
		along with other rules

		Arguments: 
			text - outputs from character recognizer CNN model concatenated to form a single word
		
		Returns: 
			corrected - correctly spelled word
		"""
		if text == "":
			return ""

		text = text.upper()

		# First check if letters are all alphabetical
		if text.isalpha():
			# Next check if the word is a valid word, if it is, don't do any spell correcting
			if self.checker.check(text):
				#print('[INFO] Detected valid word')
				return text

			# Word is probably mis-spelled, so correct it
			else:
				print('[INFO] Detected mis-spelled word')
				dic = {}
				score = 0
				for i in set(self.checker.suggest(text)):
					tmp = SequenceMatcher(None, text, i).ratio()
					dic[tmp] = i
					if tmp > score:
						score = tmp

				corrected = dic[score]

		# Check if all digits, if it is, leave it alone
		elif text.isdigit():
			#print('[INFO] Detected Number')
			return text

		elif self.checkDate(text):
			#print('[INFO] Detected Date')
			return self.processDate(text)

		elif self.checkNRIC(text):
			#print('[INFO] Detected NRIC')
			return self.processNRIC(text)

		else:
			#print('[INFO] Detected invalid word')
			candidates = self.getCandidates(text.upper())
			#print(candidates)

			# After replacing the individual characters with their commonly misclassified words, then words such as
			# "C00LB" becomes "COOLB", which we can then be corrected to a valid word, like "COOL".
			valid_words = [self.checker.suggest(x)[0] for x in candidates]
			#print(valid_words)

			# If no letters have been replaced, then candidates will be length 1, containing the original word, so just
			# return it.
			corrected = valid_words[1] if len(valid_words) >= 2 else candidates[0]

		return corrected

	def getCandidates(self, text):
		"""
		Takes a input word and suggests possible candidates of correctly spelled words by swapping out the characters
		which are commonly misclassified

		Arguments:
			text - input word

		Returns:
			candidates - a list of words where the commonly misclassified characters are swapped out
		"""
		
		candidates = [text]
		
		newtext = list(text)

		for i, letter in enumerate(text):
			if letter in self.misclassify:
				newtext[i] = self.misclassify[letter][0]
		
		if newtext != text:
			candidates.append(''.join(newtext))
		
		print(candidates)
		return candidates
	


	def checkDate(self, text):
		"""
		Simple rule-based approach to check if text matches date format e.g. 23/05/19. Note that '-' was not trained
		as one of the classes in the CNN, hence is not included for now. Future work should include '-'.
		"""
		pattern1 = re.compile(r'\d\d[ilIL/]\d\d')
		pattern2 = re.compile(r'\d[ilIL/]\d[ilIL/]')
		return re.search(pattern1, text) is not None or re.search(pattern2, text) is not None

	def processDate(self, text):
		"""
		Corrects the mis-spelled date
		"""

		corrected = text.replace('l', '/').replace('i', '/').replace('I', '/').replace('L', '/')
		#print(corrected)
		return corrected

	def checkNRIC(self, text):
		"""
		Simple rule-based approach to check if text matches NRIC format e.g. S9548422E. NRIC standard format: 
		start with either 'S' or 'G', followed by 7 numbers, then ends with a letter. For the regex, we add in some
		commonly misclassified characters as well, for instance, 'S' is commonly confused as '5', so we also check if
		the text starts with '5'.
		"""
		pattern1 = re.compile(r'^[sS5gG][0-9go]')
		return re.search(pattern1, text) is not None and text[-1].isalpha() and len(text) == 9

	def processNRIC(self, text):
		"""
		Corrects the mis-spelled NRIC
		"""

		# If starting S is confused as 5, change it
		if text[0] == '5':
			text = 'S' + text[1:]

		elif text[0].isalpha():
			text = text[0].upper() + text[1:]

		if text[-1].isalpha():
			text = text[:-1] + text[-1].upper()

		corrected = text[0] + text[1:-1].replace('g', '9').replace('s', '5').replace('L','1').replace('q', '9') + text[-1]
		#print(corrected)

		return corrected

def testSpellChecker(sc, testfile='tests.json'):
	tests = json.load(open(testfile))
	numCorrect = 0
	numWords = 0
	for k, v in tests.items():
		for mispelled in v:
			corrected = sc.correct(mispelled)
			print('[INFO] Initial: {}\t| Corrected: {}'.format(mispelled, corrected.lower()))
			numCorrect += 1 if corrected.lower() == k.lower() else 0
			numWords += 1

	print('[INFO] Percentage of Corrected Spellings: {:.2f}%'.format(numCorrect * 100 / numWords))
		

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help="input word string", default='9row')
	parser.add_argument('--t', help="supply a json file of test cases, with key: correct word and, \
								     value: list of mis-spelled words", \
								     action="store_true")
	args = parser.parse_args()

	sc = SpellCorrector()
	if args.t:
		testSpellChecker(sc)


	else:
		text = args.i
		corrected = sc.correct(text)
		print('[INFO] Initial: {}\t| Corrected: {}'.format(text, corrected))


if __name__ == "__main__":
	main()