def parse_data(data_path, file_path1, file_path2, results_path):
  if file_path1 not in ["/ABSA16_Laptops_Train_SB1_v2.xml", "/ABSA16_Restaurants_Train_SB1_v2.xml"] or \
  file_path2 not in ["/ABSA16_Laptops_Train_SB1_v2.xml", "/ABSA16_Restaurants_Train_SB1_v2.xml"]:
    raise Exception("Unknown File, you need to adjust the parsing process")

  entity_path = data_path + results_path + "-entity"
  attribute_path = data_path + results_path + "-attribute"
  res_path_entity = Path(entity_path)
  res_path_attribute = Path(attribute_path)

  try:
    res_path_entity.mkdir(parents=True,exist_ok=False)
    res_path_attribute.mkdir(parents=True,exist_ok=False)
  except FileExistsError:
    now = datetime.now()
    res_path_entity.rename(Path(entity_path + now.strftime("-%m-%d-%Y-%H-%M") + "/"))
    res_path_attribute.rename(Path(attribute_path + now.strftime("-%m-%d-%Y-%H-%M") + "/"))

    res_path_entity.mkdir(parents=True,exist_ok=False)
    res_path_attribute.mkdir(parents=True,exist_ok=False)

  # Path to save results to in Google Drive. If already exists backup old data - using Greenwich-Time
  for document in [data_path + file_path1, data_path + file_path2]:
    tree = et.parse(document)
    document_root = tree.getroot()

    ## main parsing ##
    ## root is reviews, child is review##
    for child in document_root:
      sentences = child.findall("./sentences/sentence")
      for sentence in sentences:
        text = sentence.findall("./text")[0].text
        opinions = sentence.findall("./Opinions/Opinion")
        entities = list(set([opinion.get('category').split('#')[0] for opinion in opinions]))
        attributes = list(set([opinion.get('category').split('#')[1] for opinion in opinions]))
        # If no opinion specified for sentence ignore it
        if len(entities):
          for entity in entities:
            file = open(entity_path + "/" + entity + '.txt', 'a+')
            file.write(text + '\n')
            file.close()
        
        if len(attributes):
          for attribute in attributes:
            file = open(attribute_path + "/" + attribute + '.txt', 'a+')
            file.write(text + '\n')
            file.close()

def parse_multi_label(path_to_file1, path_to_file2, path_to_save):
  try:
    Path(path_to_save).mkdir(parents=True,exist_ok=False)
  except:
    now = datetime.now()
    Path(path_to_save).rename(Path(path_to_save + now.strftime("-%m-%d-%Y-%H-%M") + "/"))
    Path(path_to_save).mkdir(parents=True,exist_ok=False)
  for document in [path_to_file1, path_to_file2]:
    tree = et.parse(document)
    document_root = tree.getroot()
    for child in document_root:
      sentences = child.findall("./sentences/sentence")
      for sentence in sentences:
        text = sentence.findall("./text")[0].text
        opinions = sentence.findall("./Opinions/Opinion")
        entities = list(set([opinion.get('category').split('#')[0] for opinion in opinions]))
        attributes = list(set([opinion.get('category').split('#')[1] for opinion in opinions]))
        # If no opinion specified for sentence ignore it
        if len(entities):
          with open(path_to_save + "/entity-text.txt", 'a+') as file:
            file.write(text + '\n')
          with open(path_to_save + "/entity-labels.txt", 'a+') as file:
            for entity in entities:
              # Use @ as delimiter between different entities
              file.write(entity + ";")
            file.write("\n")
        
        if len(attributes):
          with open(path_to_save + "/attribute-text.txt", 'a+') as file:
            file.write(text + '\n')
          with open(path_to_save + "/attribute-labels.txt", 'a+') as file:
            for attribute in attributes:
              file.write(attribute + ";")
            file.write("\n")

def load_multi_label(path_to_text, path_to_labels):
  text = []
  labels = []
  with open(path_to_text) as f_text:
    for line in f_text:
      text.append(line.strip())

  with open(path_to_labels) as f_labels:
    for line in f_labels:
      labels_in_line = []
      labels_temp = line.split(";")
      for label in labels_temp:
        if label.strip():
          labels_in_line.append(label)
      labels.append(labels_in_line)
  return text, labels