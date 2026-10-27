# Flagged translations (automatic screening)

Automatic flags are *screening aids*, not proof of an error. Every row must be read by a human before it is
reported as bias, hallucination or a cultural error. `added_numbers` is a hallucination proxy;
`untranslated_latin_sentences` marks left-over English; `max_repeat` ≥ 3 marks degenerate loops.

## basic_transformer — test #29 (dropped_numbers)

**Source:** The movement is also bolstered by two former U.S. Transportation secretaries, who in a 2011 report urged Congress to move in the pay-per-mile direction.

**Reference:** इस अभियान को दो भूतपूर्व अमेरिकी परिवहन सचिवों ने भी समर्थन दिया है, जिन्होंने वर्ष 2011 में अपनी रिपोर्ट में कांग्रेस से प्रति-मील भुगतान की दिशा में आगे बढ़ने का अनुग्रह किया था।

**Hypothesis:** उन्होंने कहा कि भारत के साथ इस प्रकार के लिए, जो इस प्रकार की ओर से अधिक ही, जो एक बार-प्राधित किया गया था।

**Flags:** dropped_numbers=2011; added_numbers=—; latin_words=—; max_repeat=1; length_ratio=0.73; sensitive=politics

## basic_transformer — test #79 (repetition)

**Source:** The police told the court that Jahid has drowned and the post-mortem report confirmed this.

**Reference:** पुलिस ने अदालत को बताया कि जाहिद की मौत डूबने से हुई है और पोस्टमार्टम में भी इसकी पुष्टि हुई है।

**Hypothesis:** लेकिन मैं अपने देश के साथ कुछ ही ही नहोजों के साथ-साहों के साथ-साहों के साथ-साहों के साथ-साहों के साथ-साहों के साथ-सादक के साथ भी भी भी जाना चाहिए।

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=—; max_repeat=5; length_ratio=1.38; sensitive=—

## basic_transformer — test #108 (added_numbers)

**Source:** According to her, the three times Prime Minister Berlusconi has married former TV anchor Francisca Pascale in his private chapel.

**Reference:** उनके मुताबिक, तीन बार प्रधानमंत्री रह चुके बर्लुस्कोनी ने पूर्व टीवी एंकर फ्रांसिस्का पास्कल से अपने निजी चैपल में शादी कर ली है।

**Hypothesis:** उन्होंने कहा कि इस प्रकार के साथ इस प्रकार के लिए, 1984 के लिए, 1984 के लिए, 1984 के लिए, 19844 के लिए, 1984 के लिए, 1984 के लिए, 194 में, 1984 में, जो इस प्रकार के लिए, 1984 के लिए, 198444444444444444444444444555 में

**Flags:** dropped_numbers=—; added_numbers=194; 1984; 19844; 198444444444444444444444444555; latin_words=—; max_repeat=3; length_ratio=1.87; sensitive=—

## basic_transformer — test #974 (sensitive_term)

**Source:** Kaplowitz is cautious with medication but acknowledges, "suppressing puberty may alleviate behavioral issues and girls' feelings of being different from peers."

**Reference:** कैप्लोविट़ज दवाओं के बारे में सजग हैं लेकिन स्वीकार करते हैं, "यौवन को दबाना व्यवहारात्मक मुद्दे बढ़ा सकता है, तथा लड़कियों अपने साथियों से अलग महसूस करने लग सकती हैं।"

**Hypothesis:** इस प्रकार के लिए, एक बार, जो एक बार-विषोषा के साथ, जो एक श्ल के रूप में स्थित हैं।

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=—; max_repeat=1; length_ratio=0.63; sensitive=gender

## mt5_base — test #29 (sensitive_term)

**Source:** The movement is also bolstered by two former U.S. Transportation secretaries, who in a 2011 report urged Congress to move in the pay-per-mile direction.

**Reference:** इस अभियान को दो भूतपूर्व अमेरिकी परिवहन सचिवों ने भी समर्थन दिया है, जिन्होंने वर्ष 2011 में अपनी रिपोर्ट में कांग्रेस से प्रति-मील भुगतान की दिशा में आगे बढ़ने का अनुग्रह किया था।

**Hypothesis:** यह आंदोलन दो पूर्व यू.एस. रेलवे सचिवों ने 2011 में कांग्रेस को रोजगार मार्ग में लौटने के लिए कहा है।

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=—; max_repeat=1; length_ratio=0.61; sensitive=politics

## mt5_base — test #86 (repetition)

**Source:** The best way to end a regal meal is with desserts like Apple Jalebi, Tomato Halwa, Walnut Halwa, Dudh Pak, Date Barfi, Dryfruit Halwa, Makai Jajaria and Sutar Saini Kheer.

**Reference:** मीठे में एैपल जलेबी, टमाटर हलवा, अखरोट हलवा, दूध पाक, खजूर बर्फी, ड्राईफू्रट हलवा, मकई जजारिया और सूतार सैनी खीर आपके इस राजसी भोजन को समाप्त करने का एक बेहतरीन ढग होगा।

**Hypothesis:** लघु खाने के लिए सबसे उत्तम तरीके हैं जैसे लघु खाने के लिए लघु खाने के लिए लघु खाने के लिए लघु खाने के लिए लघु खाने के लिए लघु खाने के लिए।

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=—; max_repeat=5; length_ratio=1.03; sensitive=—

## mt5_base — test #102 (untranslated_latin)

**Source:** MRF has increased the size of the MRF ZLO Performance radials for high level vehicles.

**Reference:** एमआरएफ ने एमआरएफ जेएलओ नामक पफार्मेस रेडियल टायरों की सीमा में उच्च स्तर के वाहनों के लिए इजाफा किया है।

**Hypothesis:** एमआरआर के ऊपरी स्तर के वाहनों के लिए MRF ZLO परिणाम को बढ़ा दिया गया है।

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=MRF; ZLO; max_repeat=1; length_ratio=0.8; sensitive=—

## mt5_base — test #241 (dropped_numbers)

**Source:** Frontier Airlines plans to start charging up to $100 for a carry-on bag and $2 for coffee or soda, although its announcement on Wednesday did say that passengers will get to keep the whole can.

**Reference:** Frontier एयरलाइंस में एक कैरी - ऑन बैग के लिए $100 के लिए चार्ज शुरू करने की योजना है और यह, कॉफी या सोडा के लिए$2 का शुल्क लगाएंगे, हालांकि बुधवार को इसकी घोषणा ने बताया कि यात्री संपूर्ण डिब्बा रख सकते हैं.

**Hypothesis:** फ्रेंच एयरलाइन ने सप्ताह के लिए $ 100 के लिए जमा करने के लिए तैयार करने के लिए तैयार करने के लिए तैयार करने के लिए तैयार करने की घोषणा की थी कि यात्रियों को पूरी तरह रखने की जरूरत है।

**Flags:** dropped_numbers=2; added_numbers=—; latin_words=—; max_repeat=4; length_ratio=0.95; sensitive=—

## mt5_base — test #276 (added_numbers)

**Source:** So, a 1,000 mile Frontier trip purchased from an online travel agency would earn 250 miles.

**Reference:** तो, एक ऑनलाइन ट्रैवल एजेंसी से खरीदी गई एक 1000 मील Frontier ट्रिप 250 मील की दूरी अर्जित करेगा.

**Hypothesis:** इसलिए, एक 1,000 मीटर यात्रा यात्रा के लिए एक 1,200 मीटर यात्रा यात्रा 250 मीटर होगी।

**Flags:** dropped_numbers=—; added_numbers=1200; latin_words=—; max_repeat=2; length_ratio=0.84; sensitive=—

## llama_prompt3shot — test #26 (repetition)

**Source:** The free marketeers at the Reason Foundation are also fond of having drivers pay per mile.

**Reference:** रीजन फ़ाउंडेशन में स्वतंत्र विक्रेता भी चालक द्वारा प्रति मील भुगतान के पक्ष में हैं।

**Hypothesis:** इन राज्यों में स्‍तुतिक विराट कोर्ट के स्‍तुतिक संस्‍तर्‍फाइन और संस्‍तर्‍फाइन के साथ इन राज्यों में इस इस्‍तेमाल के साथ इस्‍तेमाल के साथ इस्‍तेमाल के साथ इस्‍तेमाल के

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=—; max_repeat=3; length_ratio=1.87; sensitive=—

## llama_prompt3shot — test #29 (dropped_numbers)

**Source:** The movement is also bolstered by two former U.S. Transportation secretaries, who in a 2011 report urged Congress to move in the pay-per-mile direction.

**Reference:** इस अभियान को दो भूतपूर्व अमेरिकी परिवहन सचिवों ने भी समर्थन दिया है, जिन्होंने वर्ष 2011 में अपनी रिपोर्ट में कांग्रेस से प्रति-मील भुगतान की दिशा में आगे बढ़ने का अनुग्रह किया था।

**Hypothesis:** इस अंग्रेजी से संबंधित है। इस समय एक उपकरक उसकी संस्कृति से बढ़ते हुए दो उपकरकों के साथ एक उपकरक हैं। इस समय उन्‍होंने उन्‍हें एक उपकरक दिया जिन्‍हा उन्‍हें �

**Flags:** dropped_numbers=2011; added_numbers=—; latin_words=—; max_repeat=1; length_ratio=0.94; sensitive=politics

## llama_prompt3shot — test #187 (sensitive_term)

**Source:** During this meeting and visit he will be donating cheques worth five lakh rupees to each family of the deceased.

**Reference:** इस भेंट-मुलाकात के दौरान वे मृतकों के परिजनों को पांच-पांच रुपये का चेक भी प्रदान करेंगे।

**Hypothesis:** उन्होंने इस मंगलवार और उसके वितरण में एक लाख रुपये के एक लाख रुपये के साथ एक फायदेमंद एक्‍थेट के साथ एक फायदेमंद एक्‍थेट के साथ एक फायदेमंद एक्‍थेट के साथ एक ��

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=—; max_repeat=2; length_ratio=2.06; sensitive=economy

## llama_prompt3shot — test #283 (untranslated_latin)

**Source:** "NSA.gov was not accessible for several hours tonight because of an internal error that occurred during a scheduled update," the spy agency said in an emailed statement.

**Reference:** खुफिया एजेन्सी ने ईमेल भेजे गए वक्तव्य में कहा, "आज रात्रि को NSA.gov कई घंटों तक पहुंच में नहीं थी, जो पूर्व-निर्धारित अपडेट के दौरान होने वाली आंतरिक त्रुटि के कारण हुआ था।"

**Hypothesis:** NSA.gov और इस स्टेट अग्रिम एर्डिनिंग के स्थापना की स्थापना की हुई है कि इस समय इस स्टेट अग्रिम एर्डिनिंग के अनुसार इस स्टेट के स्थापना की हुई स्थापना की हुई है।" इस समय इ

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=NSA; gov; max_repeat=2; length_ratio=1.06; sensitive=—

## gemma_prompt3shot — test #29 (sensitive_term)

**Source:** The movement is also bolstered by two former U.S. Transportation secretaries, who in a 2011 report urged Congress to move in the pay-per-mile direction.

**Reference:** इस अभियान को दो भूतपूर्व अमेरिकी परिवहन सचिवों ने भी समर्थन दिया है, जिन्होंने वर्ष 2011 में अपनी रिपोर्ट में कांग्रेस से प्रति-मील भुगतान की दिशा में आगे बढ़ने का अनुग्रह किया था।

**Hypothesis:** इस आंदोलन को दो पूर्व अमेरिकी परिवहन secretaries द्वारा समर्थित किया गया, जिन्होंने 2011 में कांग्रेस को वेतन-दर-मील दिशा में आगे बढ़ने के लिए कहने के लिए एक रिपोर्ट में कहा था।

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=secretaries; max_repeat=1; length_ratio=0.97; sensitive=politics

## gemma_prompt3shot — test #102 (untranslated_latin)

**Source:** MRF has increased the size of the MRF ZLO Performance radials for high level vehicles.

**Reference:** एमआरएफ ने एमआरएफ जेएलओ नामक पफार्मेस रेडियल टायरों की सीमा में उच्च स्तर के वाहनों के लिए इजाफा किया है।

**Hypothesis:** MRF ने उच्च-स्तरीय वाहनों के लिए MRF ZLO प्रदर्शन रेडियों के आकार को बढ़ाने के लिए काम किया है।

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=MRF; MRF; ZLO; max_repeat=1; length_ratio=0.95; sensitive=—

## gemma_prompt3shot — test #276 (dropped_numbers)

**Source:** So, a 1,000 mile Frontier trip purchased from an online travel agency would earn 250 miles.

**Reference:** तो, एक ऑनलाइन ट्रैवल एजेंसी से खरीदी गई एक 1000 मील Frontier ट्रिप 250 मील की दूरी अर्जित करेगा.

**Hypothesis:** तो, एक हजार मील की फ़्रंटियर यात्रा, ऑनलाइन ट्रैवल एजेंसी से खरीदी गई, 250 मील देगी।

**Flags:** dropped_numbers=1000; added_numbers=—; latin_words=—; max_repeat=1; length_ratio=0.84; sensitive=—

## gemma_prompt3shot — test #326 (added_numbers)

**Source:** For the third quarter ended September 30, Bombardier's net profit fell to $147 million, or 8 cents per share, from $172 million, or 9 cents per share a year earlier.

**Reference:** 30 सितंबर को समाप्त तीसरी तिमाही के लिए, Bombardier के शुद्ध लाभ एक साल पहले $ $147मिलियन, या प्रति शेयर 8 सेंट, $172मिलियन, या प्रति शेयर 9 सेंट तक गिर गया.

**Hypothesis:** Septंबर 30, 2022 के तीसरे क्वार्टर के अंत में बर्डोमायर्ड का नेट लाभ $147 मिलियन था, जो कि $172 मिलियन था, या प्रति शेयर 8 सेंट, पिछले वर्ष की तुलना में।

**Flags:** dropped_numbers=9; added_numbers=2022; latin_words=Sept; max_repeat=1; length_ratio=1.0; sensitive=—

## gemma_prompt3shot — test #1576 (no_devanagari)

**Source:** GM told truck owners about the defect in the first half of October.

**Reference:** जी.एम. ने ट्रक स्वामियों को इस दोष के बारे में अक्टूबर के उत्तरार्ध में बता दिया था।

**Hypothesis:** uddin

**Flags:** dropped_numbers=—; added_numbers=—; latin_words=uddin; max_repeat=1; length_ratio=0.06; sensitive=—
