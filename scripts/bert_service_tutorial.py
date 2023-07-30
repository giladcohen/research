# run in shell:
# bert-serving-start -model_dir /data/gilad/models/uncased_L-24_H-1024_A-16/ -num_worker=1

from bert_serving.client import BertClient
from research.datasets.my_cifar100 import MyCIFAR100

bc = BertClient()
out = bc.encode(['First do it', 'then do it right', 'then do it better'])

list(bc.encode(['An airplane or aeroplane (informally plane) is a fixed-wing aircraft that is propelled forward by thrust from a jet engine, propeller, or rocket engine.'])[0])
list(bc.encode(['A car (or automobile) is a wheeled motor vehicle used for transportation.'])[0])
list(bc.encode(['Birds are a group of warm-blooded vertebrates constituting the class Aves, characterised by feathers, toothless beaked jaws, and a strong yet lightweight skeleton.'])[0])
list(bc.encode(['The cat (Felis catus) is a domestic species of a small carnivorous mammal.'])[0])
list(bc.encode(['Deer or true deer are hoofed ruminant mammals forming the family Cervidae.'])[0])
list(bc.encode(['The dog or domestic dog (Canis familiaris or Canis lupus familiaris) is a domesticated descendant of the wolf which is characterized by an upturning tail.'])[0])
list(bc.encode(['A frog is any member of a diverse and largely carnivorous group of short-bodied, tailless amphibians composing the order Anura (literally without tail).'])[0])

# CIFAR-10 description
# airplane
list(bc.encode(['An airplane is a flying vehicle that has fixed wings and engines or propellers that thrust it forward through the air.'])[0])
# automobile
list(bc.encode(['An automobile is a car: a vehicle with four wheels and an internal combustion engine.'])[0])
# bird
list(bc.encode(['A bird is an animal with wings, feathers, and two legs. Birds, from chickens to crows, are also warm-blooded and lay eggs.'])[0])
# cat
list(bc.encode(['a small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws. It is widely kept as a pet.'])[0])
# deer
list(bc.encode(['A deer is a four-legged, hoofed animal with antlers. Disney’s Bambi is a famous deer.'])[0])
# dog
list(bc.encode(['A dog is a very common four-legged animal that is often kept by people as a pet or to guard or hunt.'])[0])
# frog
list(bc.encode(['A frog is a small amphibian with long back legs that allow it to hop. Most frogs have fat little bodies and bulging eyes.'])[0])
# horse
list(bc.encode(['A horse is a large, four-legged animal with hooves, a long nose and tail, and a mane of hair along its upper back.'])[0])
# ship
list(bc.encode(["A ship is a large watercraft that travels the world's oceans and other sufficiently deep waterways, carrying goods or passengers."])[0])
# truck
list(bc.encode(['A truck is a large vehicle that is used to transport goods by road.'])[0])

# CIFAR-100
dataset = MyCIFAR100('/Users/giladcohen/data/dataset/cifar100')
for class_ in dataset.classes:
    if '_' in class_:
        strr = class_.split('_')[0] + ' ' + class_.split('_')[1]
    else:
        strr = class_
    strr = '#' + strr
    print(strr)
    print(list(bc.encode([class_])[0]))

for i, class_ in enumerate(dataset.classes[50:100]):
    if '_' in class_:
        strr = class_.split('_')[0] + ' ' + class_.split('_')[1]
    else:
        strr = class_
    strr = '#' + strr
    print(strr)
    print('bert[{}] = '.format(i+50))

# apple
list(bc.encode(['An apple is a round fruit with red or green skin and a whitish inside.'])[0])
# acuarium fish
list(bc.encode(['An aquarium fish is a creature that lives in water inside an aquarium and has a tail and fins.'])[0])
# baby
list(bc.encode(['A very young child, especially one newly or recently born, wearing a diaper and sucking a pacifier.'])[0])
# bear
list(bc.encode(['a large, heavy mammal that walks on the soles of its feet, having thick fur and a very short tail.'])[0])
# beaver
list(bc.encode(['A beaver is a furry animal with a big flat tail and large teeth. Beavers use their teeth to cut wood and build river dams.'])[0])
# bed
list(bc.encode(['A bed is a piece of furniture that you lie on when you sleep.'])[0])
# bee
list(bc.encode(['A bee is an insect with wings closely related to wasps and ants, with a yellow-and-black striped body.'])[0])
# beetle
list(bc.encode(['A beetle is a dark, shiny, hard-shelled insect.'])[0])
#bicycle
list(bc.encode(["A bicycle is a two-wheeled vehicle that's propelled by foot pedals and steered with handlebars."])[0])
#bottle
list(bc.encode(['A bottle is a glass or plastic container in which drinks and other liquids are kept.'])[0])
#bowl
list(bc.encode(['A bowl is a round dish that hold food with a wide uncovered top.'])[0])
#boy
list(bc.encode(['A male child or young man.'])[0])
#bridge
list(bc.encode(['A bridge is a structure that is built over a railway, river, or road so that people or vehicles can cross.'])[0])
#bus
list(bc.encode(['A bus is a large motor vehicle which carries passengers from one place to another.'])[0])
#butterfly
list(bc.encode(['A butterfly is an insect with large colourful wings and a thin body.'])[0])
#camel
list(bc.encode(["A camel is a four-legged desert animal with a hump on its back."])[0])
#can
list(bc.encode(['A can is a metal container in which something such as food, drink, or paint is put.'])[0])
#castle
list(bc.encode(['A castle is a large building with thick, high walls.'])[0])
#caterpillar
list(bc.encode(['A caterpillar is a small, fuzzy, worm-like animal that feeds on plants'])[0])
#cattle
list(bc.encode(['A group of animals that includes cows, buffalo, and bison, that are often kept for their milk or meat.'])[0])
#chair
list(bc.encode(['A chair is a piece of furniture for one person to sit on. Chairs have a back and four legs.'])[0])
#chimpanzee
list(bc.encode(['An African ape (animal related to monkeys) with black or brown fur.'])[0])
#clock
list(bc.encode(['A clock is an instrument, for example in a room or on the outside of a building, that shows what time of day it is.'])[0])
#cloud
list(bc.encode(['A cloud is a mass of water vapour that floats in the sky. Clouds are usually white or grey in colour.'])[0])
#cockroach
list(bc.encode(['A cockroach is a large brown insect sometimes found in the home.'])[0])
#couch
list(bc.encode(['A couch is a long, comfortable seat for two or three people.'])[0])
#crab
list(bc.encode(['A crab is a sea creature with a flat round body covered by a shell, with claws.'])[0])
#crocodile
list(bc.encode(['A crocodile is a large reptile with a long body and strong jaws. Crocodiles live in rivers and eat meat.'])[0])
#cup
list(bc.encode(['A cup is a small, round container, often with a handle, used for drinking tea, coffee, etc.'])[0])
#dinosaur
list(bc.encode(['Dinosaurs were large reptiles which lived in prehistoric times.'])[0])
#dolphin
list(bc.encode(['A dolphin is a sea mammal that is large, smooth, and grey, with a long, pointed mouth.'])[0])
#elephant
list(bc.encode(['An elephant is a very large animal with a long, flexible nose called a trunk, which it uses to pick up things.'])[0])
#flatfish
list(bc.encode(['Flatfish are sea fish with flat wide bodies, for example plaice or sole.'])[0])
#forest
list(bc.encode(['A forest is a large area where trees grow close together.'])[0])
#fox
list(bc.encode(['A fox is a wild animal which looks like a dog and has reddish-brown fur, a pointed face and ears, and a thick tail.'])[0])
#girl
list(bc.encode(['A girl is a female child.'])[0])
#hamster
list(bc.encode(['A hamster is a small furry animal which is similar to a mouse, and which is often kept as a pet.'])[0])
#house
list(bc.encode(['A house is a building in which people live, usually the people belonging to one family.'])[0])
#kangaroo
list(bc.encode(['A kangaroo is a large Australian animal which moves by jumping on its back legs.'])[0])
#keyboard
list(bc.encode(['A keyboard is the set of keys on a computer or typewriter that you press in order to make it work.'])[0])
#lamp
list(bc.encode(['A lamp is a light that works by using electricity or by burning oil or gas.'])[0])
#lawn mower
list(bc.encode(['A hand-propelled or power-driven machine for cutting the grass of a lawn.'])[0])
#leopard
list(bc.encode(['A leopard is a type of large, wild cat. Leopards have yellow fur and black spots, and live in Africa and Asia.'])[0])
#lion
list(bc.encode(['A lion is a large wild member of the cat family with yellowish-brown fur.'])[0])
#lizard
list(bc.encode(['A lizard is a small reptile that has a long body, four short legs, a long tail, and thick skin.'])[0])
#lobster
list(bc.encode(['A lobster is a sea creature that has a hard shell, two large claws, and eight legs.'])[0])
#man
list(bc.encode(['A man is an adult male human being.'])[0])
#maple tree
list(bc.encode(['A maple tree is a tree with five-pointed leaves which turn bright red or gold in autumn.'])[0])
#motorcycle
list(bc.encode(['A motorcycle is a vehicle with two wheels and an engine.'])[0])
#mountain
list(bc.encode(['A mountain is a very high area of land, much larger than a hill, with steep sides.'])[0])
#mouse
list(bc.encode(['A mouse is a small mammal with short fur, a pointed face, and a long tail.'])[0])
#mushroom
list(bc.encode(['A mushroom is a fungus with a round top and short stem. Some types of mushroom can be eaten.'])[0])
#oak tree
list(bc.encode(['An oak tree is a large tree that often grows in woods and forests and has strong, hard wood.'])[0])
#orange
list(bc.encode(['An orange in a round sweet fruit that has a thick orange skin and an orange centre divided into many parts.'])[0])
#orchid
list(bc.encode(['Orchids are plants with brightly coloured, unusually shaped flowers.'])[0])
#otter
list(bc.encode(['An otter is a small animal with brown fur, short legs, and a long tail. Otters swim well and eat fish.'])[0])
#palm tree
list(bc.encode(['Palm is a tree growing in warm regions and having a tall, straight trunk, no branches, and long pointed leaves at the top.'])[0])
#pear
list(bc.encode(['A pear is a sweet fruit, usually with a green skin and a lot of juice, that has a round base.'])[0])
#pickup truck
list(bc.encode(['A pickup truck is a small vehicle with an open part at the back in which goods can be carried.'])[0])
#pine tree
list(bc.encode(['A pine tree is an evergreen tree (one that never loses its leaves) that grows in cooler areas of the world.'])[0])
#plain
list(bc.encode(['A plain is a large area of flat land.'])[0])
#plate
list(bc.encode(['A plate is a round or oval flat dish that is used to hold food.'])[0])
#poppy
list(bc.encode(['A poppy is a plant with large, delicate flowers that are typically red and have small, black seeds.'])[0])
#porcupine
list(bc.encode(['A porcupine is an animal with a covering of long, sharp quills (stiff hairs like needles) on its back.'])[0])
#possum
list(bc.encode(['Possom, or opposum, is a small marsupial that lives in trees and has thick fur and a long nose and tail.'])[0])
#rabbit
list(bc.encode(['A rabbit is a small furry animal with long ears. Rabbits are sometimes kept as pets, or live wild in holes in the ground.'])[0])
#raccoon
list(bc.encode(['A raccoon is a small animal that has dark-coloured fur with white stripes on its face and on its long tail.'])[0])
#ray
list(bc.encode(['A ray is a fairly large sea fish which has a flat body, eyes on the top of its body, and a long tail.'])[0])
#road
list(bc.encode(['A road is a long, hard surface built for vehicles to travel along.'])[0])
#rocket
list(bc.encode(['A rocket is a space vehicle that is shaped like a long tube.'])[0])
#rose
list(bc.encode(['A rose is a flower, often with a pleasant smell, which grows on a bush with stems that have sharp points called thorns on them.'])[0])
#sea
list(bc.encode(['A sea is a large area of salty water that is part of an ocean or is surrounded by land.'])[0])
#seal
list(bc.encode(['A seal is a large mammal that eats fish and lives partly in the sea and partly on land or ice.'])[0])
#shark
list(bc.encode(['A shark is a large fish that has sharp teeth and a pointed fin on its back.'])[0])
#shrew
list(bc.encode(['A shrew is an animal like a small mouse but with a longer pointed nose and small eyes.'])[0])
#skunk
list(bc.encode(['A skunk is a small, furry, black-and-white animal with a large tail, which makes a strong, unpleasant smell as a defense when it is attacked.'])[0])
#skyscraper
list(bc.encode(['A skyscraper is a very tall building in a city.'])[0])
#snail
list(bc.encode(['A snail is a small animal with a long, soft body, no legs, and a spiral-shaped shell. Snails move very slowly.'])[0])
#snake
list(bc.encode(['A snake is a long, thin reptile without legs.'])[0])
#spider
list(bc.encode(['A spider is a small creature with eight thin legs that catches insects in a web (a net made from sticky threads).'])[0])
#squirrel
list(bc.encode(['A squirrel is a small animal covered in fur with a long tail. Squirrels climb trees and feed on nuts and seeds.'])[0])
#streetcar
list(bc.encode(['A streetcar is an electric vehicle that transports people, usually in cities, and goes along metal tracks in the road.'])[0])
#sunflower
list(bc.encode(['A sunflower is a plant, usually having a very tall stem and a single large, round, flat, yellow flower, with many long, thin, narrow petals.'])[0])
#sweet pepper
list(bc.encode(['A sweet pepper is a hollow green, red, or yellow vegetable.'])[0])
#table
list(bc.encode(['A table is a piece of furniture with a flat top that you put things on or sit at.'])[0])
#tank
list(bc.encode(['A tank is a large military fighting vehicle designed to protect those inside it from attack, driven by wheels that turn inside moving metal belts.'])[0])
#telephone
list(bc.encode(['A telephone is the piece of equipment that you use when you talk to someone by telephone.'])[0])
#television
list(bc.encode(['A television is a large box with a viewing screen which receives electrical signals and changes them into moving pictures and sound.'])[0])
#tiger
list(bc.encode(['A tiger is a large wild animal of the cat family with yellowish-orange fur with black lines that lives in parts of Asia.'])[0])
#tractor
list(bc.encode(['A tractor is a motor vehicle with large back wheels and thick tyres, used on farms for pulling machinery.'])[0])
#train
list(bc.encode(['A train is a number of carriages which are all connected together and which are pulled by an engine along a railway.'])[0])
#trout
list(bc.encode(['Trouts are any of several game fishes of the genus Salmo, related to the salmon Compare brown trout, cutthroat trout, rainbow trout.'])[0])
#tulip
list(bc.encode(['Tulips are brightly coloured flowers that grow in the spring, and have oval or pointed petals packed closely together.'])[0])
#turtle
list(bc.encode(['A turtle is a large reptile which has a thick shell covering its body and which lives in the sea most of the time.'])[0])
#wardrobe
list(bc.encode(['A wardrobe is a tall cupboard or cabinet in which you can hang your clothes.'])[0])
#whale
list(bc.encode(['A very large sea mammal that breathes air through a hole at the top of its head.'])[0])
#willow tree
list(bc.encode(['A willow tree is a tree that usually grows near water and has long, thin branches that hang down.'])[0])
#wolf
list(bc.encode(['A wolf is a wild animal that looks like a large dog.'])[0])
#woman
list(bc.encode(['A woman is an adult female human being.'])[0])
#worm
list(bc.encode(['A worm is a small animal with a long, narrow, soft body without arms, legs, or bones.'])[0])





