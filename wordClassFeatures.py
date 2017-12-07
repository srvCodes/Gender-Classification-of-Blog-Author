from __future__ import division
#from stemming.porter2 import stem 
import nltk
import re

def wordClassFeatures(text):
	home = """work home sleep today eat tired wake watch watched dinner ate bed day house tv early boring yesterday watching sit
			central familiar family native at ease rest down home homely homey"""

	conversation = """know people think person tell feel friends talk new talking mean ask understand feelings care thinking friend 
					relationship realize question answer saying"""

	family = """years aunty auntie mommy mom nephew motherhood daughter family mother father children kids parents old year child
			son sister married dad brother moved age young grandma grandpa cousin vibes three wife living college four
			high five died six baby boy spend christmas clan folk group household tribe ancestors ancestry birth brood 
			descendants descent dynasty extraction forebears genealogy generations in-laws inheritance kind kindred lineage
			network parentage pedigree progenitors progeny race relations relatives siblings sibling strain subdivision menage
			acquaintance ally associate buddy classmate mate colleague companion company cousin partner roomie roommate chum
				cohort compatriot comrade consort crony intimate pal playmate schoolmate sidekick spare well-wisher alter-ego soulmate"""

	time = """friday saturday weekend week sunday night monday tuesday thursday wednesday morning tommorrow tonight evening days
			afternoon weeks hours july busy meeting hour months month june january february march april may august september
			october november december"""

	work = """work working works job trying right met figure meet better ting try worked idea"""

	money = """money cash cheque debit credit borrow bank accountant deposit income salary savings expenditure allowance
			millionaire billionaire bonus"""

	pastAction = """said asked told looked walked called talked wanted kept took sat gave knew felt turned stopped saw ran tried
				picked left ended"""

	games = """games game team win play played playing won season beat final two hit first video second run third shot table round
			ten chance club big straight athletics pastime action amusement ball disport diversion exercise frolic gaiety picnic
			pleasure recreation """

	maleSports = """soccer volleyball cricket golf hockey pinball basketball running highjump wrestling fighting boxing chess
				rubik cube mountaineering """

	femaleSports = """ludo carrom tennis badminton ballet dancer swimming softball cross country scuba diving scuba-diving
					cross-country"""

	internet = """site email page please website web post link check blog mail information free sent comments comment using internet
				online name service dns list computer ads ad like share thanks update message subscribe upgrade tweet instagram
				facebook google snapchat gmail twitter retweet repost retweeted follow followers unfriend block spam insta tinder linkedin"""

	location = """street place town road city cities walking trip headed front car deer apartment bus area park building small
				places ride driving drive looking local sitting bar bad standing floor weather beach view gps tracking maps
				latitude longitude"""

	fun = """fun im cool mom summer awesome lol stuff pretty ill mad funny weird"""

	foodCloth = """food eating weight lunch water hair life white wearing color ice red fat body black clothes hot drink wear blue
				minutes shirt green coffee total store shopping """

	poetic = """eyes heart soul pain light deep smile dreams dark hold hands head hand alone sun dream mind cold fall air voice 
			touch blood feet words hear rain mouth"""

	books = """book read reading books story writing written movie stories movies film write character fact thoughts title short
			take wrote verse"""

	religion = """god jesus lord church earth world word lives power human believe given truth thank death evil own peace speak
				bring truly creed cult denomination myth mythology prayer ritual pray sect spirituality superstition
				theology communion observance orthodoxy pietism piety pious preference religiosity rites sacrifice sanctification
				standards veneration"""

	romance = """forget forever remember gone true face spent times love cry hurt wish loved baby babe babes darling sweetheart
				hubby wifey kiss smooch"""

	swearing = """shit fucking fuck ass bitch damn hell sucks stupid hate drunk crap kill guy gay kid sex crazy bimbo motherfucker
				asshole dildo lesbo lesbian"""

	politics = """bush obama president prime minister iraq kerry us war american political india america states country government
				john national news state support issues article michael bill report public issue history party york law major
				act fight parliament poor constitution congress"""

	music = """jazz base bass dj rj vj jockey pop cultural metal music song songs band cd rock listening listen show favourite
			radio sound heard shows sounds amazing dance"""

	school = """school teacher class study test finish english students period paper pass"""

	business = """system based products business control example personal experience general"""

	emotional = """aggressive alienated angry annoyed anxious careful cautious confused curious depressed determined disappointed 
				discouraged disgusted ecstatic embarrassed enthusiastic envious excited exhausted frightened frustrated guilty
				happy helpless hopeful hostile humiliated hurt hysterical innocent interested jealous lonely mischievous 
				miserable optimistic paranoid peaceful proud puzzled regretful relieved sad satisfied shocked shy sorry
				surprised suspicious thoughtful undecided withdrawn"""

	positive = """absolutely abundance ace active admirable adore agreed amazing appealing attraction bargain beaming beautiful
				best better boost breakthrough breeze brilliant brimming charming clean clear colorful compliment confidence
				cool courteous cuddly dazzling delicious delightful dynamic easy ecstatic efficient enhance enjoy enormous
				excellent exotic expert exquisite flair free generous genius great graceful heavenly ideal immaculate impressive
				incredible inspire luxurious outstanding royal speed splendid spectacular superb sweet sure supreme terrific
				treat treasure ultra unbeatable ultimate unique wow zest"""

	negative = """wrong stupid bad evil dumb foolish grotesque harm fear horrible idiot lame mean poor heinous hideous deficient
				petty awful hopeless fool risk immoral risky spoiled spoil malign vicious wicked fright ugly atrociuos moron
				spiteful hate meager malicious lacking """

	maleProfession = """engineer scientist physicist chemist explorer wrestler kung-fu kung fu martial arts pilot electrician
					heating mechanic technician automotive supervisor construction trades carpenter machinist truck driver
					industrial refractory machinery police officer plumber laborer maintenance worker welder jointer tiler
					firefighter chef head cook programmer surgeon physician detective astronaut cartography"""

	femaleProfession = """school teacher biologist dentist receptionist botanist kindergarten nurse medical assistant secretary
						adminstrative hairdresser hairstylist cosmetologist childcare fashion designer artist marketing advisor
						adviser lawyer judge waitress air hostess attendant retailer"""

	acronyms = """gm gn brb rofl afaik lmao lmfao lol tgif omg omfg tc ttyl af ama bae dae dm dafuq eli5 fml ftfy headdesk hifw
				icymi idgaf imo imho irl jsyk lulz mfw mrw mirl nsfw nsfl paw qtf smh tb til tbt tl yolo xoxo"""

	chastity = """sexual abstention celibacy abstenance virtue morality decency monogamy virginity abstemiousness chasteness
				cleanness continence demureness devotion honor innocence integrity modesty restraint singleness sinlessness 
				spotlessness temperance uprightness immaculacy naivete purity pureness unchaste chastity fasting frugality 
				moderation self-restraint abnegation asceticism avoidance forebearance renunciation self-control self-denial
				soberness sobriety teetolism abstaining refraining doctrine philosophy nonindulgence monasticism self-discipline
				austerity abstain diet famish hungry deny""" 

	prediction = """forecast guess indicator prognosis prophecy augury cast conjecture divination dope foresight foretelling hunch
				horoscope omen palmestry presage prevision prognostication soothsaying tip vaticination zodiac fortune-telling
				surmising"""

	envy = """hatred malice prejudice resentment rivalry backbiting coveting covetousness enviousness grudge heartburn lusting 
			malevolence malignity opposition evil-eye green-eyed grudgingness invidiousness jaundiced-eye resentfulness"""

	blueness = """blueness indecency pornography profanity smut vulgarity abomination affront atrocity bawdiness blight coarseness 
			curse dirtiness filthiness foulness immodesty impropriety impurity indelicacy lewdness licentiousness lubricity offense
			outrage porn prurience salacity scatology scurrility sleaze smuttiness smearword vileness suggestiveness x-rating"""

	pulpiness = """pulpiness curd pudding pulp pap poultice taste rob grume jam pultaceous grumous baccate"""

	sourness = """acidity asperity astringency tartness mordancy"""

	rain = """deluge drizzle flood hail mist monsoon precipitation rainfall rainstrom shower showers sleet stream torrent cloudburst
			condensation flurry pour pouring raindrops sheets spate spit sprinkle sprinkling volley drencher precip wet-stuff
			window-washer"""

	greenness = """adolescence bloom boyhood childhood girlhood ignorance immaturity inexperience juvenescence juvenility minority
			puberty prime spring springtide teens youthfulness tender-age greenness"""

	roughness = """roughness acrimony bumpiness crudity gruffness irregularity unevenness wooliness"""

	recession = """recession bankruptcy collapse decline deflation downturn hard inflation shakeout slide slump stagnation unemployment
				bust depression bad-times big-trouble bottom-out"""

	foresight = """foresight insight prudence anticipation care carefulness caution circumspection clairvoyance discernment discreetness
				discretion economy foreknowledge forethought perception precaution precognition preconception premeditation premonition
				prescience prospect providence provision sagacity canniness far-sightedness long-sightedness prenotion""" 

	evildoer = """criminal devil felon gangster lawbreaker murderer psychopath sinner sociopath troublemaker villain"""

	redness = """redness bloom blossom burning flush flushing glow glowing mantling reddening rosiness ruddiness scarlet"""

	selfishness = """selfishness greed self-indulgence self-centered self-worship stinginess"""

	virtue = """virtue advantage ethic excellence faith generosity goodness ideal kindness merit rectitude righteousness value asset
			charity consideration faithfulness fortitude high-minded hope incorrupt justice plus probity respectability 
			temper trustworthy upright worth worthy ethical ethicalness"""

	insolence = """insolence abuse arrogance audacity brass brazeness cheek chutzpah contempt contemptousness contumely effrontery
				gall guff hardihood impertinence impudence incivility insubordination lip offensiveness pertness presumption sass
				rudeness rude sauce back-talk uncivility"""

	mathematics = """mathematics algebra arithmetic geometry trigonometry calculus math maths addition division multiply multiplies
				figures barchart pie chart numbers subtraction plot graph data """

	clothing = """clothing accouterment apparel array caparison civvies costume covering drag drapery dress duds ensemble equipment
				feathers finery frippery frock garb garments gear habiliment habit garment hand-me-downs livery outfit overclothes
				panoply rags raiment regalia rigging sack sportswear tatters things threads tog trappings trousseau underclothes
				vestment vesture vines wardrobe wear full feather get-up glad rags tailleur toggery"""

	feminism = """feminism misogyny misogynist pervert rape rapist transmisogyny privilege intersectionality cisgender transgender
				radical terf swerf misandrist notallmen mansplaning manspreading womanizer paygap feminity womanhood manhood 
				insecure insecurity insecured assault molest penetrate masturbate periods pads menstruate menstruation bleed pms
				dowry alimony divorce custody patriarchy matriarchy dominated patriarchal matriarchal equality equal third-wave
				feminazi feminist meninist choice reservation bodyshaming """

	groom = """groom bridegroomm suitor benedict spouse fiance"""

	sleep = """sleep coma dream hibernate hibernation slumber trance bedtime catnap dormancy dormant doze dull dullness lethargy nap
			nod repose rest sandman shuteye siesta snooze torpidity torpor"""

	eating = """eating chewing consume consumption dine dining binging bite biting devour glutton gobbling gobble masticate munch 
			nibble overindulgence snack feast feeding feed gorging fed gourmandize meal pigging stuffing """

	selfi = """ascetic automatic autogenous automatous autonomic autonomous endogenous narcissistic narcissist subjective self """

	physical = """environmental natural real substantial concrete corporeal gross materialistic objective palpable phenomenal solid 
				somatic visible ponder ponderable sensible """

	disgust = """disgust disgusting antipathy dislike distaste loathing loathe revulsion abhorrence abominate detestation hateful
			nausea objection repugnance revolt satiation satiety sickness surfeit nauseation nauseousness"""

	fear = """chickenhearted fainthearted terrified angst anxiety concern despair dismay doubt dread horror jitters panic scare
			suspicion terror unease uneasiness worry agitation aversion awe consternation cowardice creeps discomposure 
			disquitude distress faintheartedness foreboding fright funk misgibing nightmare phobia presentiment qualm
			revulsion timidity trembling tremor trepidation bete noire chickenheartedness sweat recreancy """

	ethics = """ethics belief conduct convention honesty honor mores ethos practice principles conscience"""

	#(Schler et al., 2006)
	maleDom = "linux microsoft gaming server software gb programming google data graphics india nations democracy users economic"

	femaleDom = "shopping mom cried freaked pink cute gosh kisses yummy mommy boyfriend skirt adorable husband hubby"

	hyperlinks = "http https .com .edu .in .org www .co. "

	countHome = countConversation = countFamily = countTime = countWork = countPastAction = countAcronyms= 0
	countGames = countInternet = countLocation = countFun = countFoodCloth = countPoetic = countMoney = 0
	countBooks = countReligion = countRomance = countSwearing = countPolitics = countMalSports = countFemSports = 0
	countMusic = countSchool = countBusiness = countEmotional = countPositive = countNegative = countMalProf = countFemProf = 0
	countClothing= countMath = countRough = countGreen = countRain = countSour = 0
	countInsolence = countVirtue = countSelfish = countRedness =  countEvildo = countForesight = countRecession =  0
	countPulp = countBlue = countEnvy = countPredict = countChastity = countFeminism = 0
	countGroom = countSleep = countEating = countSelfi = countPhysical =  0
	countDisgust = countFear = countEthics =  countMalDom = countFemDom = countHyperlink = 0

	#text = re.sub('\W+',' ', text )
	totalWords = len(text.split())
	#print(totalWords)
	if totalWords is not 0:
		text = text.lower()
		text = nltk.word_tokenize(text)
		conversation = nltk.word_tokenize(conversation)
		home = nltk.word_tokenize(home)
		family = nltk.word_tokenize(family)
		time = nltk.word_tokenize(time)
		pastAction = nltk.word_tokenize(pastAction)
		work = nltk.word_tokenize(work)
		games = nltk.word_tokenize(games)
		internet = nltk.word_tokenize(internet)
		location = nltk.word_tokenize(location)
		fun = nltk.word_tokenize(fun)
		foodCloth = nltk.word_tokenize(foodCloth)
		poetic = nltk.word_tokenize(poetic)
		books = nltk.word_tokenize(books)
		religion = nltk.word_tokenize(religion)
		romance = nltk.word_tokenize(romance)
		swearing = nltk.word_tokenize(swearing)
		politics = nltk.word_tokenize(politics)
		music = nltk.word_tokenize(music)
		school = nltk.word_tokenize(school)
		business = nltk.word_tokenize(business)
		maleSports = nltk.word_tokenize(maleSports)
		femaleSports = nltk.word_tokenize(femaleSports)
		positive= nltk.word_tokenize(positive)
		negative = nltk.word_tokenize(negative)
		maleProfession = nltk.word_tokenize(maleProfession)
		femaleProfession = nltk.word_tokenize(femaleProfession)
		emotional = nltk.word_tokenize(emotional)
		money = nltk.word_tokenize(money)
		acronyms = nltk.word_tokenize(acronyms)
		chastity = nltk.word_tokenize(chastity)
		prediction = nltk.word_tokenize(prediction)
		envy = nltk.word_tokenize(envy)
		blueness = nltk.word_tokenize(blueness)
		pulpiness = nltk.word_tokenize(pulpiness)
		sourness = nltk.word_tokenize(sourness)
		rain = nltk.word_tokenize(rain)
		greenness = nltk.word_tokenize(greenness)
		roughness = nltk.word_tokenize(roughness)
		recession = nltk.word_tokenize(recession)
		foresight = nltk.word_tokenize(foresight)
		evildoer = nltk.word_tokenize(evildoer)
		redness = nltk.word_tokenize(redness)
		selfishness = nltk.word_tokenize(selfishness)
		virtue= nltk.word_tokenize(virtue)
		insolence = nltk.word_tokenize(insolence)
		mathematics = nltk.word_tokenize(mathematics)
		clothing= nltk.word_tokenize(clothing)
		feminism = nltk.word_tokenize(feminism)
		groom = nltk.word_tokenize(groom)
		sleep = nltk.word_tokenize(sleep)
		eating = nltk.word_tokenize(eating)
		selfi = nltk.word_tokenize(selfi)
		physical = nltk.word_tokenize(physical)
		disgust = nltk.word_tokenize(disgust)
		fear = nltk.word_tokenize(fear)
		ethics = nltk.word_tokenize(ethics)
		maleDom = nltk.word_tokenize(maleDom)
		femaleDom = nltk.word_tokenize(femaleDom)
		#hyperlinks = nltk.word_tokenize(hyperlinks)

		for word in text:
			if word in conversation:
				countConversation += 1

			if word in home:
				countHome += 1

			if word in family:
				countFamily += 1

			if word in time:
				countTime += 1

			if word in pastAction:
				countPastAction += 1

			if word in work:
				countWork += 1

			if word in games:
				countGames += 1

			if word in internet:
				countInternet += 1

			if word in location:
				countLocation += 1

			if word in fun:
				countFun += 1

			if word in foodCloth:
				countFoodCloth += 1

			if word in poetic:
				countPoetic += 1

			if word in books:
				countBooks += 1

			if word in religion:
				countReligion += 1

			if word in romance:
				countRomance += 1

			if word in swearing:
				countSwearing += 1

			if word in politics:
				countPolitics += 1

			if word in music:
				countMusic += 1

			if word in school:
				countSchool += 1

			if word in business:
				countBusiness += 1

			if word in emotional:
				countEmotional += 1
			
			if word in positive:
				countPositive += 1
			
			if word in negative:
				countNegative += 1

			if word in maleSports:
				countMalSports += 1

			if word in femaleSports:
				countFemSports += 1

			if word in maleProfession:
				countMalProf += 1

			if word in femaleProfession:
				countFemProf += 1

			if word in money:
				countMoney += 1

			if word in acronyms:
				countAcronyms += 1

			if word in chastity:
				countChastity += 1

			if word in prediction:
				countPredict += 1

			if word in envy:
				countEnvy += 1

			if word in blueness:
				countBlue += 1

			if word in pulpiness:
				countPulp += 1

			if word in sourness:
				countSour += 1

			if word in rain:
				countRain += 1

			if word in greenness:
				countGreen += 1

			if word in roughness:
				countRough += 1

			if word in recession:
				countRecession += 1

			if word in foresight:
				countForesight += 1

			if word in evildoer:
				countEvildo += 1

			if word in redness:
				countRedness += 1

			if word in selfishness:
				countSelfish += 1

			if word in virtue:
				countVirtue += 1

			if word in insolence:
				countInsolence += 1

			if word in mathematics:
				countMath += 1

			if word in clothing:
				countClothing += 1

			if word in feminism:
				countFeminism += 1

			if word in groom:
				countGroom += 1

			if word in sleep:
				countSleep += 1

			if word in eating:
				countEating += 1

			if word in selfi:
				countSelfi += 1

			if word in physical:
				countPhysical += 1

			if word in disgust:
				countDisgust += 1

			if word in fear:
				countFear += 1
					
			if word in ethics:
				countEthics += 1

			if word in maleDom:
				countMalDom += 1

			if word in femaleDom:
				countFemDom += 1

		#	for item in hyperlinks:
			#	if re.search(word, item) or re.search(item, word):
				#	countHyperlink += 1	

		countFeminism /= 1.0*totalWords
		countConversation /= 1.0*totalWords
		countHome /= 1.0*totalWords
		countFamily /= 1.0*totalWords
		countTime /= 1.0*totalWords
		countPastAction /= 1.0*totalWords
		countWork /= 1.0*totalWords
		countGames /= 1.0*totalWords
		countInternet /= 1.0*totalWords
		countLocation /= 1.0*totalWords
		countFun /= 1.0*totalWords
		countFoodCloth /= 1.0*totalWords
		countPoetic /= 1.0*totalWords
		countBooks /= 1.0*totalWords
		countReligion /= 1.0*totalWords
		countRomance /= 1.0*totalWords
		countSwearing /= 1.0*totalWords
		countPolitics /= 1.0*totalWords
		countMusic /= 1.0*totalWords
		countSchool /= 1.0*totalWords
		countBusiness /= 1.0*totalWords
		countEmotional /= 1.0*totalWords
		countPositive /= 1.0*totalWords
		countNegative /= 1.0*totalWords
		countMalProf /= 1.0*totalWords	
		countFemProf /= 1.0*totalWords
		countMalSports /= 1.0*totalWords
		countFemSports /= 1.0*totalWords
		countMoney /=1.0*totalWords
		countAcronyms /=1.0*totalWords
		countChastity /= 1.0*totalWords
		countPredict /= 1.0*totalWords
		countEnvy /= 1.0*totalWords
		countBlue /= 1.0*totalWords
		countPulp /= 1.0*totalWords
		countSour /= 1.0*totalWords
		countRain /= 1.0*totalWords
		countGreen /= 1.0*totalWords
		countRough /= 1.0*totalWords
		countRecession /= 1.0*totalWords
		countForesight /= 1.0*totalWords
		countEvildo /= 1.0*totalWords
		countRedness /= 1.0*totalWords
		countSelfish /= 1.0*totalWords
		countVirtue /= 1.0*totalWords
		countInsolence /= 1.0*totalWords
		countMath /= 1.0*totalWords
		countClothing /= 1.0*totalWords
		countGroom /= 1.0*totalWords
		countSleep /= 1.0*totalWords
		countEating /= 1.0*totalWords
		countSelfi /= 1.0*totalWords
		countPhysical /= 1.0*totalWords
		countDisgust /= 1.0*totalWords
		countFear /= 1.0*totalWords
		countEthics /= 1.0*totalWords
		countMalDom /= 1.0*totalWords
		countFemDom /= 1.0*totalWords
		#countHyperlink /= 1.0*totalWords

		return(countHome,countConversation,countFamily,countTime,countWork,countPastAction,
		countGames,countInternet,countLocation,countFun,countFoodCloth,countPoetic,
		countBooks,countReligion,countRomance,countSwearing,countPolitics, countMalDom, countFemDom,
		countMusic,countSchool,countBusiness,countEmotional, countPositive ,countNegative ,
		countMalProf, countFemProf, countMalSports, countFemSports, countMoney, countAcronyms,
	    countClothing, countMath, countInsolence, countVirtue, countSelfish, countRedness, 
		countEvildo, countForesight, countRecession, countRough, countGreen, countRain,
		countSour, countPulp, countBlue, countEnvy, countPredict, countFeminism, countGroom, countSleep,
		countEating, countSelfi, countPhysical, countDisgust, countFear, countChastity, countEthics)

	else:
		return (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
		0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

text = """This is hopeless countless priceless and I am indecisive. so sorry sorry I am feeling terrible 
that I am unable to fulfil a WONderful TV mathematical brutal vicious terrific problem."""
print(wordClassFeatures(text))
