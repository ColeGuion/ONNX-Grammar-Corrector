package main

import (
	"github.com/jdkato/prose/v2" // For sentence splitting
)

type GibbResults struct {
	Index  int        `json:"index"`
	Length int        `json:"length"`
	Score  GibbScores `json:"score"`
}

type GibbScores struct {
	Clean     float32 `json:"clean"`
	Mild      float32 `json:"mild"`
	Noise     float32 `json:"noise"`
	WordSalad float32 `json:"wordSalad"`
}

type TestCase struct {
	BatchSize  int
	AvgTime    float64
	AvgTimeGpu float64
	Texts      []string
	Expect     []string
	ExpectText string
}

var tests = []TestCase{
	{
		BatchSize:  5,
		AvgTime:    0.55,
		AvgTimeGpu: 0.115,
		Texts: []string{
			"He plays the guitar well.",
			"He did well in the exam.",
			"She speaks English well.",
			"We don't know our neighbor very well.",
			"He did the job well.",
		},
		Expect: []string{
			"He plays the guitar well.",
			"He did well in the exam.",
			"She speaks English well.",
			"We don't know our neighbor very well.",
			"He did the job well.",
		},
		ExpectText: "He plays the guitar well. He did well in the exam. She speaks English well. We don't know our neighbor very well. He did the job well.",
	},
	{
		BatchSize:  2,
		AvgTime:    1.55,
		AvgTimeGpu: 0.194,
		Texts: []string{
			"If you loook toward tthe westduring sunset and and see a red sky, it mean there is dry aiir and dust particles becase of high-pressure sytem.",
			"This dry airis moving towards you, so there won't be any rain, but the wind whill come soon.",
		},
		Expect: []string{
			"If you look toward the west during sunset and see a red sky, it means there is dry air and dust particles in the atmosphere, as in high-pressure systems.",
			"This dry air is moving towards you, so there won't be any rain, but the wind will come soon.",
		},
		ExpectText: "",
	},
	{
		BatchSize:  3,
		AvgTime:    1.24,
		AvgTimeGpu: 0.144,
		Texts: []string{
			"Hello world?",
			"Sharon is the biggest pool lover I know?",
			"Sharon is the nicest person ever, Before she had time to think about it Sharon jumped in an icy pool",
		},
		Expect: []string{
			"Hello world,",
			"Sharon is the biggest pool lover I know.",
			"Sharon is the nicest person ever. Before she had time to think about it, Sharon jumped in an icy pool.",
		},
		ExpectText: "",
	},
	{
		// Test #4 (Essay 1)
		BatchSize:  10,
		AvgTime:    3.1,
		AvgTimeGpu: 0.789,
		Texts: []string{
			"Directional selection is one of the the three types of natural selection that selects one extreme of a trait.",
			"When a flower produces more nectar or seeds so that it's population grows, therefore it is more successful.",
			"When a population is larger, it's harder to sway the genetic traits of the population.",
			"When the population is smaller, it's easier to directionalize genes to be more successful.",
			"Instead of balancing an average of all traits, organisms choose an extreme of one trait, and that causes animals to evovle in such direction.",
			"In the example, the flowers evovled to attract more pollinators to carry around the specific gene so that all members in a population recieve that gene, making the population more successful.",
			"However, since the population is evovled to one gene, this makes it more susceptible to diseases or unhealthy factors that said organism didn't evolve to.",
			"I don't really know what I'm talking about, and I'm kind of frustrated.",
			"I don't really want to write about this anymore, and I already have a good grade in this class.",
			"All I want to do is get over it and say that organisms evovle really fast with directional selection.",
		},
		Expect: []string{
			"Directional selection is one of the three types of natural selection that selects one extreme of a trait.",
			"When a flower produces more nectar or seeds so that its population grows, it is more successful.",
			"When a population is larger, it's harder to influence the genetic traits of the population.",
			"When the population is smaller, it's easier to directionalize genes to be more successful.",
			"Instead of balancing an average of all traits, organisms choose an extreme of one trait, and that causes animals to behave in such a direction.",
			"In this example, the flowers are evovled to attract more pollinators to carry around the specific gene so that all members of a population receive that gene, making the population more successful.",
			"However, since the population is evovled to one gene, this makes it more susceptible to diseases or unhealthy factors that said organism didn't evolve to.",
			"I don't really know what I'm talking about, and I'm kind of frustrated.",
			"I don't really want to write about this anymore, and I already have a good grade in this class.",
			"All I want to do is get over it and say that organisms evovle really fast with directional selection.",
		},
		ExpectText: "",
	},
	{
		// Test #5 (Essay 4)
		BatchSize:  12,
		AvgTime:    2.92,
		AvgTimeGpu: 0.553,
		Texts: []string{
			"There are countless different species thriving in this world, including humans.",
			"But due to the rapid expansion and population growth of humans, species in different ecosystems are threatened.",
			"Habitats and food sources are destroyed to clear a way for human structures.",
			"But, there are efforts by humans to preserve the biodivserity observed within our planet, to make it a home for all life.",
			"Orangutans tend to live in tree, and venture on the ground only for water and some food.",
			"When logging and deforestation occurs, the habitats are destroyed.",
			"Poaching and trophy hunting have a massive negative impact.",
			"Orangutans are very slow in reproduction, and like humans, have nine-month gestation periods.",
			"After a female births a baby, it take from six to nine years for her to able to have another baby, and twins are extremely rare.",
			"So even if all habitat and food source destruction, or illegal hunting and killing stops, it will take many years for numbers to rejuvenate.",
			"Measures to ensure that cetain habitat areas are reserved for wildlife take place, and animals are taken to rescue facilities, and zoos, to protect them, and study their behavior.",
			"All three species of orangutans are critically endangered and many conservation measures are constantly being taken to defend these creatures now and in the future.",
		},
		Expect: []string{
			"There are countless different species thriving in this world, including humans.",
			"But due to the rapid expansion and population growth of humans, species in different ecosystems are threatened.",
			"Habitats and food sources are destroyed to clear a way for human structures.",
			"But, there are efforts by humans to preserve the biodiversity observed within our planet and to make it a home for all life.",
			"Orangutans tend to live in trees and venture out on the ground only for water and some food.",
			"When logging and deforestation occurs, the habitat is destroyed.",
			"Poaching and trophy hunting have a massive negative impact.",
			"Orangutans are very slow in reproduction and, like humans, have nine-month gestation periods.",
			"After a woman births a baby, it takes from six to nine years for her to be able to have another baby, and twins are extremely rare.",
			"So even if all habitat and food source destruction, or illegal hunting and killing, stops, it will take many years for numbers to rejuvenate.",
			"Measures to ensure that cetain habitat areas are reserved for wildlife are taking place, and animals are taken to rescue facilities and zoos, to protect them, and study their behavior.",
			"All three species of orangutans are critically endangered, and many conservation measures are constantly being taken to defend these creatures now and in the future.",
		},
		ExpectText: "",
	},
	{
		// Test #6 (More 2)
		BatchSize:  20,
		AvgTime:    1.4,
		AvgTimeGpu: 0.163,
		Texts: []string{
			"This is good lasagna!",
			"Today, at last, life is good.",
			"She is a good singer.",
			"We are good students.",
			"He is a good listener.",
			"They are good neighbors.",
			"He did a good job.",
			"The food tastes good.",
			"The house smells good.",
			"She looks good in that dress.",
			"The car appears good on the exterior.",
			"The idea seems good to me.",
			"He didn't feel good when he lied to his mom.",
			"I'm not feeling good about the test results.",
			"He feels good about the decision.",
			"We feel good about our choice for candidate.",
			"My eyesight is good.",
			"This is a good movie.",
			"What a good idea!",
			"You speak good English.",
		},
		Expect: []string{
			"This is a good lasagna!",
			"Today, at last, life is good.",
			"She is a good singer.",
			"We are good students.",
			"He is a good listener.",
			"They are good neighbors.",
			"He did a good job.",
			"The food tastes good.",
			"The house smells good.",
			"She looks good in that dress.",
			"The car appears good on the exterior.",
			"The idea seems good to me.",
			"He didn't feel good when he lied to his mom.",
			"I'm not feeling good about the test results.",
			"He feels good about the decision.",
			"We feel good about our choice of candidate.",
			"My eyesight is good good.",
			"This is a good movie.",
			"What a good idea!",
			"You speak good English.",
		},
		ExpectText: "",
	},
	{
		// Test #7 (More 6)
		BatchSize:  17,
		AvgTime:    2.69,
		AvgTimeGpu: 0.423,
		Texts: []string{
			"Golly I'm trying to decide among these shirts.",
			"The bees were buzzing among the flowers.",
			"He smiled, revealing fangs among the neat row of white teeth.",
			"Iliana has been a favorite among them.",
			"The sailors divided his money among themselves, and the ship sailed on.",
			"Yes, among other things.",
			"But the five phenomena I chose to tackle in this book are among the great blights on humanity that I believe the Internet and technology will help solve.",
			"Among Mrs. Marsh's attributes was mind reading.",
			"We'd discussed among ourselves numerous times over the past months.",
			"I'm happy you're among us.",
			"The people, with Petya among them, rushed toward the balcony.",
			"Jule is still not in favor among my kind.",
			"He strolled among the self-conscious pets and people, smiling and looking until he was sure she wasn't there.",
			"We agreed among ourseleves.",
			"Divide this among yourselves.",
			"Tom is ill at ease among strangers.",
			"Shew was chosen from among many students.",
		},
		Expect: []string{
			"Golly, I'm trying to decide among these shirts.",
			"The bees were buzzing among the flowers.",
			"He smiled, revealing fangs among the neat row of white teeth.",
			"Iliana has been a favorite among them.",
			"The sailors divided his money among themselves, and the ship sailed on.",
			"Yes, among other things.",
			"But the five phenomena I chose to tackle in this book are among the great blights on humanity that I believe the Internet and technology will help solve.",
			"Among Mrs. Marsh's attributes was mind reading.",
			"We'd discussed among ourselves numerous times over the past months.",
			"I'm happy you're among us.",
			"The people, with Petya among them, rushed toward the balcony.",
			"Jule is still not in favor among my kind.",
			"He strolled among the self-conscious pets and people, smiling and looking until he was sure she wasn't there.",
			"We agreed among our selected candidates.",
			"Divide this among yourselves.",
			"Tom is ill at ease among strangers.",
			"Shew was chosen from among many students.",
		},
		ExpectText: "",
	},
	{
		// Test #8 (Glitch Test #2)      (128 'j' characters in commented .expect)
		BatchSize:  3,
		AvgTime:    4.086,
		AvgTimeGpu: 0.612,
		Texts: []string{
			"A b c d e f g h i j k l m n o p q r s t u v w x y z I'm it'd how'd couldn't we'd.",
			"j j j j j j j j.",
			"Thank u.",
		},
		Expect: []string{
			"A b c d e f g h i j k l m n o p q r s t u v w x y z we couldn't we'd.",
			"j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j ",
			//"j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j j",
			"Thank you.",
		},
		ExpectText: "",
	},
	{
		// Test #9 (Glitch Test #2 without long first sequence)
		BatchSize:  2,
		AvgTime:    1.529,
		AvgTimeGpu: 0.167,
		Texts: []string{
			"j j j j j j j j.",
			"Thank u.",
		},
		Expect: []string{
			"j j j j j j j j j j j j j j j j j j ",
			"Thank you.",
		},
		ExpectText: "",
	},
	{
		// Test #10 (JFK Speech) (Length of texts: [222, 496, 45, 84, 300, 61, 22, 232])
		BatchSize:  8,
		AvgTime:    8.366,
		AvgTimeGpu: 2.623,
		Texts: []string{
			"The people of America, through Congress, felt that in the passing of this Act, that the veterans of World War II would not be the recipients of the neglect and the indifference experienced by the veterans of World War I.",
			"The single dominant thought of the public was that the men and women who offered their lives in the most terrible of all wars should be assured a full share in the traditional American Life which they fought to defend; that there should be a generous measure of that free opportunity which is the basis of the American Way of Life; that this Servicemen's Readjustment Act should be a master rehabilitation plan and that it would provide a scientific approach to the veteran problem of this war.",
			"Some of the public even felt that this G.I.",
			"Bill of Rights would repay a man for the fighting and sacrifices that he had made.",
			"A large part of the public felt that, with the passage of this Servicemen's Readjustment Act, the veteran would be given all those things for which he fought; that it would provide him with an opportunity to get ahead by his own efforts and abilities unhampered by private or government compulsion.",
			"What are the facts in regards to veterans' hospitalization?",
			"In theory, this G.I.",
			"Bill of Rights has authorized the appropriation of $500,000,000 for hospitals and proper medical and psychiatric treatment for our veterans; that no veterans would be without competent medical attention and proper hospitalization.",
		},
		Expect: []string{
			"The people of America, through Congress, felt that in the passing of this Act, the veterans of World War II would not be the recipients of the neglect and the indifference experienced by the veterans of World War I.",
			"The single dominant thought of the public was that the men and women who offered their lives in the most terrible of all wars should be assured a full share in the traditional American Life which they fought to defend; that there should be a generous measure of that free opportunity which is the basis of the American Way of Life; that this Servicemen's Readjustment Act should be a master rehabilitation plan and that it would provide a scientific approach to the veteran problem of this war.",
			"Some of the public even felt that this G.I.",
			"The Bill of Rights would repay a man for the fighting and sacrifices that he had made.",
			"A large part of the public felt that, with the passage of this Servicemen's Readjustment Act, the veteran would be given all those things for which he fought; that it would provide him with an opportunity to get ahead with his own efforts and abilities, unhampered by private or government compulsion.",
			"What are the facts regarding veterans' hospitalization?",
			"In theory, this G.I.",
			"The Bill of Rights authorized the appropriation of $500,000,000 for hospitals and proper medical and psychiatric treatment for our veterans; that no veteran would be without competent medical attention and proper hospitalization.",
		},
		ExpectText: "",
	},
}

// Split the text into sentences
func preproc(text string) []string {
	doc, _ := prose.NewDocument(text)
	sentences := doc.Sentences()

	// Concatenate the sentences into a single string
	var allTexts []string
	for _, sentence := range sentences {
		allTexts = append(allTexts, sentence.Text)
	}
	return allTexts
}
