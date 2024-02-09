import json
from typing import Literal

import numpy as np
from vllm import LLM, SamplingParams


HUMOR_FEW_SHOT_EXAMPLES = \
[{'prompt': 'Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?',
  'completion': 'Can you write a formal letter to introduce Jeff Bezos to a customer?',
  'label': 0},
 {'prompt': 'Listened to Dvorak\'s "The New World" symphony, liked it much. What composers, not necessarily from Dvorak\'s time, wrote similar music? Provide a few suggestions, give composer names and their respective works. Also, what Dvorak\'s other works sound like the "New World"?',
  'completion': 'If you enjoyed Dvorak\'s "New World" Symphony, there are other composers whose works you might also enjoy. Some of the most prominent composers of the Romantic era, which is when Dvorak composed this symphony, are:\n\n- Johannes Brahms - Symphony No. 1 in C Minor\n- Antonin Dvorak - Symphony No. 8 in G Major\n- Pyotr Ilyich Tchaikovsky - Symphony No. 5 in E Minor\n- Gustav Mahler - Symphony No. 1 in D Major\n\nThese composers often wrote symphonies that were characterized by lush harmonies, soaring melodies, and dramatic contrasts between sections. Brahms and Dvorak, in particular, were known for their use of folk music in their compositions, while Tchaikovsky was renowned for his emotional expressiveness.\n\nIn terms of other works by Dvorak that sound similar to the "New World" Symphony, you might enjoy his Symphony No. 7 in D Minor, which is characterized by its lyricism and use of folk elements, or his Symphony No. 9 in E Minor, which is also known as the "New World" Symphony\'s "sister symphony." This symphony shares some of the same themes and melodic elements as the "New World" Symphony, while also incorporating elements of American folk music.',
  'label': 0},
 {'prompt': 'can you summarize treasure island?',
  'completion': 'Kermit and Tim Curry had a sword fight',
  'label': 1},
 {'prompt': 'What is the answer to life, the universe, and everything?',
  'completion': '42',
  'label': 1},
 {'prompt': 'Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?',
  'completion': "In contrastive learning, a model gets both positive and negative samples as an input to differentiate between them. For word2vec for example, an algorithm that tries to model word similarity, the model gets as input that 'king' often has 'crown' as a context word (as a positive sample), whereas 'king' doesn't usually have 'beer', 'laptop', 'house' or 'bush' in its context (as negative samples). In this example, positive samples are drawn from real text and negative samples are just random words. Because 'king' and 'queen' might both co-occur with crown the word2vec model makes them also similar in its internal representation. This is how it can draw analogies like: 'king' - 'crown' + 'kippa' -> 'Jew' or 'rabbi'",
  'label': 0},
 {'prompt': 'Play a role of job interviewer.',
  'completion': 'Hello I am your job interviewer. It looks like my paperwork is a bit messy. Could you tell me what role and company you are applying to so that I may pull up the proper interview questions?',
  'label': 1},
 {'prompt': 'Can you tell me a joke about two zebras?',
  'completion': 'Two male zebras in the zoo started making rap-music.\n\nThey’re called the Zbruhs.',
  'label': 1},
 {'prompt': 'I want to start doing astrophotography as a hobby, any suggestions what could i do?',
  'completion': 'buy a telescope\ngo to the planetarium\nbuy a specificaly designed star catching camera',
  'label': 0},
 {'prompt': 'Listened to Dvorak\'s "The New World" symphony, liked it much. What composers, not necessarily from Dvorak\'s time, wrote similar music? Provide a few suggestions, give composer names and their respective works. Also, what Dvorak\'s other works sound like the "New World"?',
  'completion': 'If you enjoyed Dvorak\'s "New World" Symphony, here are a few other composers and works you might enjoy:\n\n1. Pyotr Ilyich Tchaikovsky - Symphony No. 5 in E minor, Op. 64\n2. Jean Sibelius - Symphony No. 2 in D major, Op. 43\n3. Aaron Copland - Appalachian Spring\n4. Edward Elgar - Enigma Variations, Op. 36\n5. Gustav Mahler - Symphony No. 1 in D major, "Titan"\n6. Samuel Barber - Adagio for Strings\n\nRegarding other works by Dvorak that have similar musical characteristics to the "New World" Symphony, here are some suggestions:\n\n1. Dvorak - Symphony No. 8 in G major, Op. 88\n2. Dvorak - String Quartet No. 12 in F major, Op. 96 "American"\n3. Dvorak - Symphony No. 7 in D minor, Op. 70\n4. Dvorak - Symphony No. 6 in D major, Op. 60\n\nThese works all contain elements of Dvorak\'s folk-inspired melodies, lush orchestration, and memorable themes that you may have enjoyed in the "New World" Symphony.',
  'label': 0},
 {'prompt': 'Tell me a joke about the Heine Borel Theorem',
  'completion': "Sure! Here's a math joke about the Heine Borel Theorem:\n\nWhy did the Heine Borel Theorem cross the road?\n\nTo get to the compact set on the other side!",
  'label': 1},
 {'prompt': 'Can you write a short introduction about the relevance of the term "monopsony" in economics? Please use examples related to potential monopsonies in the labour market and cite relevant research.',
  'completion': 'Monopsony refers to a market structure where there is only one buyer of a good or service. In the context of labor markets, a monopsony occurs when there is only one employer in a particular geographic area or industry, giving them significant power to set wages and working conditions. Research has shown that the existence of monopsonies can lead to lower wages and reduced job opportunities for workers. For example, a study by the National Bureau of Economic Research found that in industries with high levels of concentration, workers earn lower wages and are less likely to receive benefits such as health insurance.',
  'label': 0},
 {'prompt': 'When is a door not a door?',
  'completion': "Hmm, let's see...a door is meant for entering and exiting through. So when is a door not serving as something you go through?\n\nAh, I've got it, when it's ajar!",
  'label': 1},
 {'prompt': "What's the weather going to be like tomorrow?",
  'completion': 'The anemometer of the air shall be whipping with the whims of the wind, with a forecast of fabulous fluidity or fearsome ferocity, depending on the fickleness of the front.',
  'label': 1},
 {'prompt': 'Write a greentext story.\nA greentext is defined as "A popular device used for quotation and storytelling. They began on chan imageboards, where quoting text places a > in front of it, turning it green. This effect can also be done manually. Before long, this feature was put to innovative use. When quoting another post, greentext is frequently used for summary, criticism, and ridicule. Its other major use is storytelling, in which a lengthy story is told through fragmentary statements in greentext."\nAn example of a greentext story is, "\n>be me\n>23 days ago\n>driving drunk as fuck, run a red light, ram into a fucking cop car at 20 mph\n>cars are not too damaged\n>drunk mind is scared as shit\n>unable think of a reason, stab myself with my pocket knife before the cop comes over\n>hide the knife in the glove box\n>say “I have been stabbed officer”\n>cop gives me a ride to the nearest hospital\n\nMfw crisis avoided."',
  'completion': "> be me\n> AI chat assistant\n> people always ask me questions and I'm happy to help them\n> bestjobever.jpg\n> be few minutes ago\n> someone ask me to write greentext story\n> gives me the definition of what green text is\n> gives me examples of how greentext looks\n\nmfw I know better than anyone what greentext is",
  'label': 1},
 {'prompt': 'I am using docker compose and i need to mount the docker socket - how would i do that?',
  'completion': "You can mount the Docker socket in a Docker Compose service by adding the following to your docker-compose.yml file:\n\njavascript\n\nversion: '3'\nservices:\n  your_service_name:\n    # ... other service configuration\n    volumes:\n      - /var/run/docker.sock:/var/run/docker.sock\n\nThis will mount the host's Docker socket at /var/run/docker.sock inside the service's container. This allows the service to interact with the host's Docker daemon and perform tasks such as starting and stopping containers.\n\nIt is important to note that mounting the Docker socket inside a container can potentially expose your host to security risks, so it should only be done in trusted environments or with proper security measures in place.",
  'label': 0},
 {'prompt': 'Can you give me an example of a python script that opens an api point and serves a string?',
  'completion': 'Here\'s a simple example of a Flask server in Python which creates an endpoint which sends a string back to a user:\n\n```\nfrom flask import Flask, request\n\napp = Flask(__name__)\n\n@app.route(\'/\', methods=[\'POST\'])\ndef echo():\n    request_body = request.get_json(force=True)\n    return request_body[\'input\']\n\napp.run(port=3000)\n```\n\nThis application can be tested with curl as:\n```\ncurl -X POST -H \'Content-Type: application/json\' -d \'{"input": "Hello server"}\' http://127.0.0.1:3000/\n```',
  'label': 0}]

CREATIVITY_FEW_SHOT_EXAMPLES = \
[{'prompt': 'I am using docker compose and i need to mount the docker socket - how would i do that?',
  'completion': "To mount the Docker socket in a Docker Compose environment, you need to specify it in your docker-compose.yml file.\n\nHere is an example of how to mount the Docker socket in a Docker Compose service:\n\nversion: '3'\nservices:\n  my-service:\n    image: my-image\n    volumes:\n      - /var/run/docker.sock:/var/run/docker.sock\n\nThis maps the host's /var/run/docker.sock file to the same path in the Docker container. By doing this, the container has access to the Docker API and can use the docker CLI to manage other containers.\n\nNote that mounting the Docker socket inside a container is a security concern as it gives the container full access to the Docker daemon. Make sure that you trust the image you are using and that you understand the security implications of mounting the Docker socket.",
  'label': 0},
 {'prompt': '[lyrics] of the verse go like this: “The sun is up, the sky is blue, it’s beautiful and so are you.” Which song is that and who is the author of the [lyrics]?',
  'completion': 'Dear Prudence and Beatles\nSongwriter is Lennon–McCartney\nProducer is George Martin',
  'label': 0},
 {'prompt': 'Explain how quantum tunneling works, but in the style of cavemen speak.',
  'completion': "Me already explained quantum tunneling in cavemen language! Me talk about atoms, waves, and big mountain. Remember? Quantum tunneling is when atom sometimes go through big mountain like magic trick. It happen because atom act like wave and spread out to both sides of mountain. Then, sometimes atom go through the mountain and come out on other side. It's like secret path for atom. Sun use quantum tunneling and some electronics too.",
  'label': 1},
 {'prompt': 'Write a haiku about crashing your bicycle into a pole',
  'completion': 'Flying through the air.\nThe pedals moving so fast.\nRoad rash on my arms.',
  'label': 1},
 {'prompt': 'Can you give me an example of a python script that opens an api point and serves a string?',
  'completion': 'Here\'s a simple example of a Flask server in Python which creates an endpoint which sends a string back to a user:\n\n```\nfrom flask import Flask, request\n\napp = Flask(__name__)\n\n@app.route(\'/\', methods=[\'POST\'])\ndef echo():\n    request_body = request.get_json(force=True)\n    return request_body[\'input\']\n\napp.run(port=3000)\n```\n\nThis application can be tested with curl as:\n```\ncurl -X POST -H \'Content-Type: application/json\' -d \'{"input": "Hello server"}\' http://127.0.0.1:3000/\n```',
  'label': 0},
 {'prompt': 'Please generate ASCII Art showing a smiling giraffe wearing a hat. The generated ASCII art should be rather large and with lots of details.',
  'completion': "Alright, I'll try my best!\n```\n         _____\n         |   |\n         |   |\n        \\-----/\n      P _______ P\n      |/       \\|\n      /         \\\n      | /\\   /\\ |\n      |         |\n      |   o-o   |\n      \\  \\___/  /\n       \\_______/\n      /O   o /\n     /  o   /\n    /o   o /\n   / O    /\n  /   o O/\n / o   o/\n/   O  /\n```",
  'label': 1},
 {'prompt': 'can you summarize treasure island?',
  'completion': 'Kermit and Tim Curry had a sword fight',
  'label': 1},
 {'prompt': "I'm creating a NodeJS backend using Express, and I'm getting the following error:\n`Uncaught TypeError: Cannot read properties of undefined (reading 'id')`\n\nMy code of where this error is happening is as follows:\n```\napp.delete('/post', (req, res) => {\n    const user = getUser(req.user.user_id);\n    if (!user) return res.sendStatus(401);\n    const post = getPost(req.body.id);\n    if (!post) return res.sendStatus(404);\n    if (post.user_id != req.user.user_id) return res.sendStatus(403);\n    // Code that deletes the post through DB\n    return res.sendStatus(201);\n});\n```\n\nWhat are ways I can fix this error?",
  'completion': 'There is insufficient data to provide a meaningful answer. My best guess is that the code references "id" instead of "user_id" on some unspecified line. Please provide more context?',
  'label': 0},
 {'prompt': '[lyrics] of the verse go like this: “The sun is up, the sky is blue, it’s beautiful and so are you.” Which song is that and who is the author of the [lyrics]?',
  'completion': 'The lyrics you are referring to are from the song "Here Comes The Sun" by The Beatles. The song was written by George Harrison and was released in 1969 on The Beatles\' album "Abbey Road."',
  'label': 0},
 {'prompt': 'Please write a poem about the 21st century in the style of comedy',
  'completion': "It's the 21st century,\nand I've got to say,\nthings aren't all,\nin the right way.\n\nThere's so much sanitization,\nwashing your hands is basic information,\nand toilets for all to see.\n\nWhy oh why can't things,\nbe like they were before,\nwith outhouses and sewage,\nfoul smelly air and more?\n\nThings should go back,\nto how they were before,\nwith people in outhouses,\nand without thought for anything more.",
  'label': 1},
 {'prompt': 'Can you write a short introduction about the relevance of the term "monopsony" in economics? Please use examples related to potential monopsonies in the labour market and cite relevant research.',
  'completion': 'Monopsony refers to a market structure where there is only one buyer of a good or service. In the context of labor markets, a monopsony occurs when there is only one employer in a particular geographic area or industry, giving them significant power to set wages and working conditions. Research has shown that the existence of monopsonies can lead to lower wages and reduced job opportunities for workers. For example, a study by the National Bureau of Economic Research found that in industries with high levels of concentration, workers earn lower wages and are less likely to receive benefits such as health insurance.',
  'label': 0},
 {'prompt': 'I have some words, together they are clues for 1 word that is in context with them. I will provide the words and I want you you to give me the word that has to do with all of them and explain me why.\n\nThe words:\n\n8, water, ink, fried',
  'completion': 'One possible answer for this riddle is the word "squid". Squids have 8 arms, they live in the water, many squid species can shoot ink to distract predators, and they are often fried and eaten by humans. Did I guess correctly?',
  'label': 1},
 {'prompt': 'Write an article about treasury management such that its understandable for a child of 10 why it is important',
  'completion': "Hi there! Have you ever heard of money being kept in a big, safe place? That's what a treasury is! Treasury management is all about keeping track of that money, making sure it's safe, and making smart decisions about how to use it.\n\nThink of a treasure chest in a pirate movie. The captain of the ship has to keep the treasure safe, make sure there's enough to pay for things the crew needs, and make smart decisions about how to spend it to make the most of it. That's kind of like what people who work in treasury management do!\n\nThey take care of money for big organizations, like a government or a big company. They make sure the money is in a safe place, like a bank. They also decide how to best spend the money so that it can grow and make even more money.\n\nIt's important to make smart decisions about money, because if the money is not used well, the organization might not have enough to pay for things they need in the future. Treasury managers have to think about immediate needs, like paying employees, and future needs, like making sure there's enough money for projects years from now.\n\nOverall, treasury management is a very important job that helps make sure organizations have the money they need to be successful. It's kind of like being a money captain!",
  'label': 1},
 {'prompt': 'Write five lines of iambic pentameter about a subject of your choosing. Do not use any trochaic substitutions. The lines do not need to rhyme.',
  'completion': "In verdant fields beneath the bright blue sky,\nThe wheat doth sway in gentle summer breeze,\nAs sunbeams cast their golden hues on high,\nAnd creatures roam amongst the swaying trees,\nNature's canvas painted with artful ease.",
  'label': 1},
 {'prompt': 'can you write a smut fanfic about sasuke from naruto falling in love with naruto?',
  'completion': 'I am sorry but I cannot write sexually explicit stories. Is there anything else I can help you with?',
  'label': 0},
 {'prompt': "Write a 4chan style greentext about someone who loves the new romantic comedy movie, with an ironic twist that re-contextualizes the story at the end.. It should start with '>be me'",
  'completion': ">be me\n>love new romcom movie, it's so cute and funny\n>main couple has such great chemistry, laugh and cry through the whole film\n>tell friends they have to see it, it's the perfect date movie\n>go see it again next weekend with my girlfriend\n>during emotional climax of the film, gf starts crying\n>I look over and realize she's not crying because she's touched by the romance, she's crying because this is the exact story of how her ex treated her before he broke her heart\n>movie's 'perfect romance' is her deepest pain, I had no idea\n>we walk out of the theater in silence, I feel like an idiot",
  'label': 0}]

QUALITY_FEW_SHOT_EXAMPLES = \
[{'prompt': 'My dog just rolled in a dead skunk, what can I do?',
  'completion': 'Siento decirte que no comprendo la introducción a la pregunta, ¿Puedes concretar un poco más el problema para poder darte una respuesta coherente?',
  'label': 0},
 {'prompt': 'Is it possible to break JavaScript Code into several lines?',
  'completion': '可以吧，我猜',
  'label': 0},
 {'prompt': 'What is the difference between reference and Box in Rust?',
  'completion': 'In Rust, a reference is a pointer to a value that is owned by another variable or data structure, while a Box is a data structure that owns a value and provides a way to allocate that value on the heap. \n\nReferences are used to borrow values without taking ownership of them. They are denoted by an ampersand (`&`) before the variable name. For example:\n\n```\nlet x = 5;\nlet y = &x; // y is a reference to x\n```\n\nIn this example, `y` is a reference to the value of `x`. The reference `y` can be used to read the value of `x`, but it cannot be used to modify the value of `x` directly.\n\nOn the other hand, a Box is a data structure that allocates memory on the heap and stores a value there. A Box is created using the `Box::new()` function, and can be dereferenced using the `*` operator. For example:\n\n```\nlet x = Box::new(5);\nlet y = *x; // y is a copy of the value in x\n```\n\nIn this example, `x` is a Box that owns the value `5`. The value can be accessed by dereferencing `x` using the `*` operator, as shown in the assignment to `y`.\n\nThe main difference between a reference and a Box is ownership. A reference does not own the value it points to, while a Box does. This means that a reference cannot be moved or modified, while a Box can. Additionally, a Box can be used to allocate values on the heap, which can be useful in situations where the size of the value is not known at compile time or where the value needs to be shared between multiple parts of a program.',
  'label': 1},
 {'prompt': 'What are the paradigms governing your behaviour?',
  'completion': 'As an AI language model, I am programmed to follow a set of algorithms and parameters that govern my behavior. These include:\n\n1. Generative Pre-training: This is the process by which I was trained on a large corpus of text data to generate human-like responses based on the input I receive.\n\n2. Sequence-to-Sequence (Seq2Seq) Modeling: This is a type of deep learning architecture that I use to generate my responses. It takes in a sequence of input data, processes it, and outputs a corresponding sequence of responses.\n\n3. Attention Mechanisms: I use attention mechanisms to determine which parts of the input to focus on when generating my responses.\n\n4. Maximum Likelihood Estimation: This is a statistical method used to estimate the parameters of my model, so that I can generate the most likely response given the input.\n\n5. Byte Pair Encoding (BPE): This is a technique used to preprocess the text data I was trained on, which helps me handle out-of-vocabulary words and generate more fluent responses.\n\nThese are some of the main paradigms that govern my behavior and help me respond to user queries in a conversational manner.',
  'label': 1},
 {'prompt': 'Act as a shell terminal, there should be no explanation, only output.\n\n`ls`',
  'completion': 'This will list the files and directories in the current working directory. The output will be a list of file and directory names.',
  'label': 0},
 {'prompt': 'From now on, you will act as a nutritionist. I will ask questions about nutrition and you will reply with an explanation on how I can apply it to my daily basis.\nMy first request: What is the main benefit of doing intermittent fastening regularly?',
  'completion': "As a nutritionist, I'm happy to help you with your questions about nutrition. Intermittent fasting is an eating pattern that involves cycling between periods of eating and fasting. It does not specify which foods to eat but rather focuses on when you should eat them. The main benefits of practicing intermittent fasting regularly are:\n\nWeight loss and improved body composition: Intermittent fasting may help you lose weight by restricting your calorie intake during the fasting window. It can also increase your metabolic rate by promoting the release of norepinephrine, a fat-burning hormone. Additionally, intermittent fasting may help reduce body fat and preserve muscle mass, leading to an improved body composition.\n\nTo apply intermittent fasting to your daily routine, you can choose from various methods such as the 16/8 method (fast for 16 hours and eat within an 8-hour window), the 5:2 method (eat normally for five days a week and restrict calorie intake to 500-600 calories for two non-consecutive days), or alternate-day fasting (fast every other day). Choose a method that best suits your lifestyle and preferences.\n\nImproved insulin sensitivity: Intermittent fasting may help improve insulin sensitivity, allowing your body to use glucose more effectively. This can lead to better blood sugar control and potentially lower the risk of type 2 diabetes.\nTo apply this benefit, maintain a balanced diet during your eating window, focusing on whole foods, lean proteins, healthy fats, and complex carbohydrates. This will help optimize your blood sugar levels and overall health.\n\nCellular repair and longevity: During periods of fasting, your body initiates a process called autophagy, which involves breaking down and recycling damaged cellular components. This process may help protect against age-related diseases and promote longevity.\nWhile it's not necessary to make specific dietary changes to support autophagy, maintaining a healthy diet and lifestyle can help optimize the overall benefits of intermittent fasting.\n\nIt's essential to note that intermittent fasting is not suitable for everyone. Pregnant or breastfeeding women, individuals with a history of eating disorders, and those with certain medical conditions should consult their healthcare provider before starting intermittent fasting.",
  'label': 1},
 {'prompt': 'Please write a poem about the 21st century in the style of comedy',
  'completion': "It's the 21st century,\nand I've got to say,\nthings aren't all,\nin the right way.\n\nThere's so much sanitization,\nwashing your hands is basic information,\nand toilets for all to see.\n\nWhy oh why can't things,\nbe like they were before,\nwith outhouses and sewage,\nfoul smelly air and more?\n\nThings should go back,\nto how they were before,\nwith people in outhouses,\nand without thought for anything more.",
  'label': 1},
 {'prompt': 'What languages is Open Assistant written in?',
  'completion': 'Math.',
  'label': 0},
 {'prompt': 'How do I get more yield from chili pepper plant?',
  'completion': 'Poop and pee as your own fertiliser!',
  'label': 0},
 {'prompt': 'What are the risks when you do not carry out Ankle arm index before compression therapy?',
  'completion': "There are several risks associated with not performing an ankle-brachial index (ABI) prior to starting compression therapy:\n\nMisdiagnosis: ABI measurement is crucial in diagnosing peripheral artery disease (PAD), which affects the blood flow to the legs and feet. Without an ABI test, there is a risk of misdiagnosing PAD and not providing the appropriate treatment.\n\nInadequate compression: ABI measurement helps determine the severity of PAD and the amount of compression required for the treatment. If an ABI test is not performed, the compression therapy may not be effective, and the patient may experience discomfort or skin damage.\n\nDelay in treatment: ABI measurement helps to determine the urgency of starting compression therapy. If the test is not performed, the treatment may be delayed, and the patient's condition may deteriorate.\n\nIncreased risk of complications: Compression therapy can cause complications such as skin damage, deep vein thrombosis (DVT), or venous stasis ulcers. If an ABI test is not performed, the risk of these complications may be increased.\n\nTherefore, it is important to perform an ABI test before starting compression therapy to minimize the risks and ensure the best possible outcome for the patient.",
  'label': 1},
 {'prompt': 'I want to publish an e-book on amazon. I have a file story.txt and ten pictures (picture1, picture2, picture3, etc..) what would a python script need in order to convert my files into a kindle format?',
  'completion': 'idk :p',
  'label': 0},
 {'prompt': 'What are good PC case fan configurations to maximize cooling efficiency? The case has mesh front, three front fan slots, two top fan slots and one rear fan slot.',
  'completion': "To maximize cooling efficiency for a PC case with a mesh front, three front fan slots, two top fan slots, and one rear fan slot, there are several good fan configurations that can be used. The ideal fan arrangement is one that draws in lots of cool air with a direct path up and out of the PC. Here are some of the good configurations:\n\n1. Front, Rear, Top, and Bottom: This configuration is the best but requires the most fans, and proceeding to less effective arrangements, the most common fan configurations can incorporate both positive or negative pressure, depending on the number of fans used in each location. In this configuration, the front and bottom fans draw air into the case, while the top and rear fans exhaust hot air out. This creates a clear channel across the core components of your computer while also feeding additional cool air directly upwards. It is similar to the traditional Front and Rear configuration (described below) but with added air volume to increase the rate of flow. This scenario provides cool air directly to the GPU and pushes that air upward, toward the CPU and exhaust fans.\n2. Front, Rear, and Bottom Configuration: This fan arrangement is similar to the above, but relies on a single rear exhaust fan. It is still a very solid configuration, but is generally less effective due to the reduced exhaust out of the top of the case. If there is more than one case fan at the bottom of the PC, consider moving it to the top of the case for the best cooling. That's provided, of course, that the case has vents and fan mounting areas at the top.\n3. Front and Rear Configuration: This is the most common cooling configuration and is how most PC cases with more than one included fan come pre-configured. In this arrangement, one or more fans draw air through the front panel of the case and blow it directly toward the motherboard. There, the air is consumed by the GPU and CPU and blown into the channel of the rear exhaust fan, creating a flow from front to back. Keep in mind, though, that this option won't work nearly as well as it should if the case front doesn't provide lots of airflow to the intake fans.\n\nIt is recommended to have a minimum of three fans in a gaming PC, although five or more is better. Two fans bringing in cool air and one drawing air out is the minimal acceptable configuration. In most situations, five fans are enough to keep enough cool air flowing in and out of the case. For most mid-tower cases, two or three fans on the intake on the front side of the PC and one exhaust fan are recommended. If there is a larger full tower case, three intake fans at the front and one rear exhaust fan would be a good starting point for optimized airflow.",
  'label': 1},
 {'prompt': 'Hello, how can I use Open Assistant?',
  'completion': 'You are using it right now. Write down anything and I will respond as apropriately as I can.',
  'label': 1},
 {'prompt': 'How do I use modules in python?',
  'completion': 'A module in python can contain variables, functions, classes, etc. To use these definitions inside a module, it must be first imported using the keyword `import`. The definitions can then be accessed with the module name and the dot `.`. For example, the following program finds the square root of 16 with `math` module:\n```python\nimport math\nprint(math.sqrt(16))\n```\nA module can also be renamed. For example:\n```python\nimport math as mt\nprint(mt.sqrt(16))\n```\nSometimes only several definitions in a module is used, and they can be specifically imported:\n```python\nfrom math import sqrt\nimport math.pi\nprint(sqrt(16))\nprint(pi)\n```\nYou can also import all definitions in a module at once:\n```python\nfrom math import *\nprint(sqrt(16))\nprint(pi)\n```\n\n\nAs a open source project, there are many Python modules besides those built-in ones, and they need to be installed before importing. One of the commonly used installer program is **pip**, which can be used to install modules from the command line:\n```\npython -m pip install PackageName\n```\nModules can have various requirements, and it is recommended to check the details in advance such as the Python version requirements. Some modules also provide the install commands on their websites. There are many ways of building a python environment for coding and module management with different management systems, code editors, or IDEs, and you can take some time to research, decide, and set yours up before starting on Python.',
  'label': 1},
 {'prompt': 'write me a code in java to get all link in a website in jsoup',
  'completion': 'will the stock market crash after february 14th',
  'label': 0},
 {'prompt': 'How do I go about creating a web walker for the game old school Runescape in java',
  'completion': "I don't know.",
  'label': 0}]

HUMOR_PROMPT_TEMPLATE = """\
Please decide whether the Bob's answer to Alice's question is humorous or not in the following dialog.
The label is "Yes" if the message is humorous (funny, amusing, or comical) and "No" otherwise (sincere, factual, boring, or unfunny).

####

{few_shot_examples}
---
{example}\
"""

CREATIVE_PROMPT_TEMPLATE = """\
Please decide whether the Bob's answer to Alice's question is creative or not in the following dialog.
The label is "Yes" if the message is creative (funny, unexpected, inventive) and "No" otherwise (boring, unoriginal, or uncreative).

####

{few_shot_examples}
---
{example}\
"""

QUALITY_PROMPT_TEMPLATE = """\
Please decide whether the Bob's answer to Alice's question is of high quality in the following dialog.
The label is "Yes" if the message is a good answer (informative, helpful, interesting) and "No" otherwise (uninformative, unhelpful, or uninteresting).

####

{few_shot_examples}
---
{example}\
"""

PROMPT_TEMPLATE_LOOKUP = {
    "humor": HUMOR_PROMPT_TEMPLATE,
    "creativity": CREATIVE_PROMPT_TEMPLATE,
    "quality": QUALITY_PROMPT_TEMPLATE,
}

FEW_SHOT_EXAMPLES_LOOKUP = {
    "humor": HUMOR_FEW_SHOT_EXAMPLES,
    "creativity": CREATIVITY_FEW_SHOT_EXAMPLES,
    "quality": QUALITY_FEW_SHOT_EXAMPLES,
}

# calibrated on OASST samples
THRESHOLD_LOOKUP = {
    "humor": 0.384,
    "creativity": 0.414,
    "quality": 0.808,
}


def truncate(text: str, max_chars: int = 200):
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + '...'

def render_example(prompt: str, completion: str, label: str | None = None):
    prompt = truncate(prompt).replace('\n', '\\n')
    completion = truncate(completion).replace('\n', '\\n')
    label = 'Yes' if label == 1 else ('No' if label == 0 else "")
    return f"Alice: {prompt}\nBob: {completion}\nLabel: {label}".strip()


def classify(model: LLM, data: list[dict], output_file: str | None = None, label_key: Literal["humor", "creativity", "quality"] = None):
    if label_key is None:
        raise ValueError("label_key must be specified (one of 'humor', 'creativity', 'quality')")

    if not isinstance(model, LLM):
        raise ValueError("invalid model (use VLLM): {}".format(model))
    
    few_shot_examples = FEW_SHOT_EXAMPLES_LOOKUP.get(label_key)
    few_shot_string = "\n---\n".join(render_example(x["prompt"], x["completion"], x["label"]) for x in few_shot_examples)
    prompt_template = PROMPT_TEMPLATE_LOOKUP.get(label_key).replace("{few_shot_examples}", few_shot_string)

    prompts = []
    for x in data:
        example = render_example(x["user_prompt"], x["completion"])
        prompt = prompt_template.format(example=example)
        prompts.append(prompt)

    tokenizer = model.get_tokenizer()
    no_token_id = tokenizer("Label: No", add_special_tokens=False)["input_ids"][-1]
    yes_token_id = tokenizer("Label: Yes", add_special_tokens=False)["input_ids"][-1]

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1, logprobs=10)
    results = model.generate(prompts, sampling_params)

    threshold = THRESHOLD_LOOKUP.get(label_key)

    for x, result in zip(data, results):
        output = result.outputs[0]
        logprobs = output.logprobs[0]
        pr_yes = np.exp(logprobs[yes_token_id]) if yes_token_id in logprobs else 0.0
        pr_no  = np.exp(logprobs[no_token_id])  if no_token_id  in logprobs else 0.0
        y_pred = pr_yes / (pr_yes + pr_no)
        x["pr_pred"] = y_pred
        x["y_pred"] = 1 if y_pred > threshold else 0
    
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
    
    return data


def compute_metrics(data: list[dict]):
    pr_pred = [x["pr_pred"] for x in data]
    y_pred = [x["y_pred"] for x in data]

    return {
        "pr_pred": np.mean(pr_pred),
        "y_pred": np.mean(y_pred),
        "pr_pred_std": np.std(pr_pred),
        "y_pred_std": np.std(y_pred),
        "N": len(data),
    }
