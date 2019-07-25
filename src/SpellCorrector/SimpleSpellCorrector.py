import enchant
import json
import re
import argparse

class SimpleSpellCorrector:
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
		self.misclassify = json.load(open('misclassify.json'))

	def correct(self, text):
		"""
		Applies PyEnchant's spell checking algorithm: https://faculty.math.illinois.edu/~gfrancis/illimath/windows/aszgard_mini/movpy-2.0.0-py2.4.4/manuals/PyEnchant/PyEnchant%20Tutorial.htm
		along with other rules

		Arguments: 
			text - outputs from character recognizer CNN model concatenated to form a single word
		
		Returns: 
			corrected - correctly spelled word
		"""

		# First check if letters are all alphabetical
		if text.isalpha():
			text = text.lower()

			# Next check if the word is a valid word, if it is, don't do any spell correcting
			if self.checker.check(text):
				return text

			# Word is probably mis-spelled, so correct it
			else:
				corrected = self.checker.suggest(text)[0]

		# Check if all digits, if it is, leave it alone
		elif text.isdigit():
			return text

		elif self.checkDate(text):
			return self.processDate(text)

		elif self.checkNRIC(text):
			return self.processNRIC(text)

		else:
			candidates = self.getCandidates(text)
			valid_words = [x for x in candidates if self.checker.check(x)]
			# for now, without a language model, we arbitraily pick the first valid word in the list as the corrected word because
			# the list is already ordered according to confusion matrix i.e. "9" is most commonly confused for "g" then "q"
			if len(valid_words) != 0:
				corrected = valid_words[0]

			# If no valid words can be found, then just return the original word
			else:
				return candidates[0]

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
		for i, char in enumerate(text):
			if char in self.misclassify:
				for newchar in self.misclassify[char]:
					candidate = text[:i] + newchar + text[i+1:]
					candidates.append(candidate)

		return candidates

	def checkDate(self, text):
		"""
		Simple rule-based approach to check if text matches date format e.g. 23/05/19. Note that '-' was not trained
		as one of the classes hence is not included for now. Future work should include '-'.
		"""
		pattern1 = re.compile(r'\d\d[ilIL/]\d\d')
		pattern2 = re.compile(r'\d[ilIL/]\d')
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

		


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help="input word string", default='9row')
	args = parser.parse_args()

	sc = SimpleSpellCorrector()
	text = args.i
	corrected = sc.correct(text)

	print('\n[INFO] Initial Text: {}'.format(text))
	print('[INFO] Corrected Text: {}'.format(corrected))



if __name__ == "__main__":
	main()