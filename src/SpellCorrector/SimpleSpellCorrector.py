import enchant

class SimpleSpellCorrector:
	def __init__(self, lexicon="lexicon.txt"):
		if lexicon is None:
			self.checker = enchant.Dict("en_US")
		else:
			self.checker = enchant.DictWithPWL("en_US", lexicon)

	def correct(self, text):
		"""
		Applies PyEnchant's spell checking algorithm: https://faculty.math.illinois.edu/~gfrancis/illimath/windows/aszgard_mini/movpy-2.0.0-py2.4.4/manuals/PyEnchant/PyEnchant%20Tutorial.htm

		Arguments: 
			text - a single word
		
		Returns: 
			correctly spelled word
		"""

		if self.checker.check(text):
			return text

		else:
			corrected = self.checker.suggest(text)[0]
			print('[INFO] Initial Text: {}'.format(text))
			print('[INFO] Corrected Text: {}'.format(corrected))

		return corrected
		

if __name__ == "__main__":
	sc = SimpleSpellCorrector()
	sc.correct('khoi')