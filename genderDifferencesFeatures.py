import nltk

def genderDifferencesFeatures(text):

	groom = """cleaner clean washer wash perfume shave shaved shampoo cleansing soap shower
	toothpaste cream facewash moisturizer nail lipstick makeup"""

	sleep = """tiresome sleeping dazed sleeps insomnia napping nap siesta nightmare dream dreams bed
	pillow"""

	i = """me myself mine my i"""

	eating = """fat dinner tasting drunken fed breakfast cookie eat tasted skinny cookbook"""

	disgust = """sickening revolting horror sick offensive obscene nauseous wicked"""

	fear = """suspense creep dismay fright terrible terror afraid scare alarmed panicked panic"""

	sewing = """mending stiching knitting knitter knit mend tailor suture embroidery seamstress needle"""

	purpleness = """purple mauve magenta lilac lavender orchid violet mulberry purply"""

	sweetness = """syrup honey sugar bakery nectar sweet frost sugary dessert glaze nut"""

	brownness = """coffee biscuit biscuits walnut rust berry brown brunette cinnamon mahogany caramel chocolate"""

	chastity = """shame elegant decent virtue virgin delicate faithful faithfulness platonic purity spotless"""

	relig = """bless satanism angel communion spirit lord immortal theology prayers"""

	metaph = """suicide meditation cemetary temples drained immortalized mercy mourning"""

	tv = """show ad comedies comedy tv actors drama soaps video theatres commercials commercial film films"""

	job = """credentials department financials desktop manage employee work career"""

	oponent = """finalist rival enemy competitor foe opposite defendant player dissident"""

	theology = """creed scholastic religious secularism theology religion divine faith dogma"""

	uniformity = """evenness constancy constant persistence accordance steadiness steady firm firmness stable stability"""

	engineering = """automotive process industrial manufacture measure construction technician"""

	influence = """power force weak weakness inflexible ineffective charisma charm wimpy"""

	
	countGroom = countSleep = countI = countEating = countDisgust = countFear = countSewing = 0
	countPurpleness = countSweetness = countBrownness = countChastity = countRelig = countInfluence = 0
	countMetaph = countTV = countJob = countOponent = countTheology = countUniformity = countEngineering = 0

	totalWords = len(text.split())
	#print(totalWords)

	text = text.lower()
	text = nltk.word_tokenize(text)
	groom = nltk.word_tokenize(groom)
	sleep = nltk.word_tokenize(sleep)
	i = nltk.word_tokenize(i)
	eating = nltk.word_tokenize(eating)
	disgust = nltk.word_tokenize(disgust)
	fear = nltk.word_tokenize(fear)
	sewing = nltk.word_tokenize(sewing)
	purpleness = nltk.word_tokenize(purpleness)
	sweetness = nltk.word_tokenize(sweetness)
	brownness = nltk.word_tokenize(brownness)
	chastity = nltk.word_tokenize(chastity)
	relig = nltk.word_tokenize(relig)
	influence = nltk.word_tokenize(influence)
	metaph = nltk.word_tokenize(metaph)
	tv = nltk.word_tokenize(tv)
	job = nltk.word_tokenize(job)
	oponent = nltk.word_tokenize(oponent)
	theology = nltk.word_tokenize(theology)
	uniformity = nltk.word_tokenize(uniformity)
	engineering = nltk.word_tokenize(engineering)
	

	for word in text:
		if word in groom:
			countGroom += 1

		if word in sleep:
			countSleep += 1

		if word in i:
			countI += 1

		if word in eating:
			countEating += 1

		if word in disgust:
			countDisgust += 1

		if word in fear:
			countFear += 1

		if word in sewing:
			countSewing += 1

		if word in purpleness:
			countPurpleness += 1

		if word in sweetness:
			countSweetness += 1

		if word in brownness:
			countBrownness += 1

		if word in chastity:
			countChastity += 1

		if word in relig:
			countRelig += 1

		if word in metaph:
			countMetaph += 1

		if word in tv:
			countTV += 1

		if word in job:
			countJob += 1

		if word in oponent:
			countOponent += 1

		if word in theology:
			countTheology += 1

		if word in uniformity:
			countUniformity += 1

		if word in engineering:
			countEngineering += 1

		if word in influence:
			countInfluence += 1

	try:
		countGroom /= 1.0 * totalWords
	except:
		countGroom = 0
	try:
		countSleep /= 1.0 * totalWords
	except:
		countSleep = 0
	try:
		countI /= 1.0
	except:
		countI  = 0
	try:
		countEating /= 1.0 * totalWords
	except:
		countEating = 0
	try:
		countDisgust /= 1.0 *totalWords
	except:
		countDisgust = 0
	try:
		countFear /= 1.0 * totalWords
	except:
		countFear = 0
	try:
		countSewing /= 1.0 * totalWords
	except:
		countSewing = 0
	try:
		countPurpleness /= 1.0 * totalWords
	except:
		countPurpleness = 0
	try:
		countBrownness /= 1.0 * totalWords
	except:
		countBrownness = 0
	try:
		countSweetness /= 1.0 * totalWords
	except:
		countSweetness = 0
	try:
		countChastity /= 1.0 * totalWords
	except:
		countChastity = 0
	try:
		countRelig /= 1.0 * totalWords
	except:
		countRelig = 0
	try:
		countMetaph /= 1.0 * totalWords
	except:
		countMetaph = 0
	try:
		countJob /= 1.0 * totalWords
	except:
		countJob = 0
	try:
		countTV /= 1.0 * totalWords
	except:
		countTV = 0
	try:
		countOponent /= 1.0 * totalWords
	except:
		countOponent = 0
	try:
		countTheology /= 1.0 * totalWords
	except:
		countTheology = 0
	try:
		countUniformity /= 1.0 * totalWords
	except:
		countUniformity = 0
	try:
		countEngineering /= 1.0 * totalWords
	except:
		countEngineering = 0
	try:
		countInfluence /= 1.0 * totalWords
	except:
		countInfluence = 0

	return(countGroom, countSleep, countI, countEating, countDisgust, countFear, countSewing, countPurpleness,
		countSweetness, countBrownness, countChastity, countRelig, countMetaph, countJob, countTV, countOponent,
		countTheology, countUniformity, countEngineering, countInfluence)

text = """This is hopeless countless priceless and I am indecisive. so sorry sorry I am feeling terrible 
that I am unable to fulfil a WONderful TV mathematical brutal vicious terrific problem."""
print(genderDifferencesFeatures(text))
