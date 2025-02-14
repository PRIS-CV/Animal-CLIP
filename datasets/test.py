def run_clip(image_input, label):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/16', device)

    # prepare the inputs
    # val_file_name = '/home/zhangruxu/AnimalKingdom/annotation/fewer_action&category_image/animal_kingdom_origin_image_val_getFrame.xlsx'
    # data = pd.read_excel(val_file_name, sheet_name='Sheet1')
    # data = [getdata[i] for i in range(len(getdata))]

    actions_file_name = '/home/zhangruxu/AnimalKingdom/annotation/fewer_action&category_image/labels.xlsx'
    getaction = pd.read_excel(actions_file_name).Action
    action = [getaction[i] for i in range(len(getaction))]

    # image = []
    # # label = [[] for i in range(len(data))]
    # for j in range(len(data)):
    #     for i in range(len(data.columns)):
    #         column_x = data.iloc[j, [i]]
    #         if i == 0:
    #             temp = (column_x.values)[0].split(' ')
    #             image.append(temp[3])
    #             # label[j].append(action[int(temp[4])])
    #         if i != 0:
    #             if isinstance((column_x.values)[0], str):
    #                 temp = (column_x.values)[0].split(' ')
    #                 # label[j].append(action[int(temp[0])])

    # assert len(image) == len(label)

    # Draw the input tensor: labels, using it as input of function get_map
    # get_map_labels = torch.zeros((len(image), len(action)))
    # for i in range(len(image)):
    #     for j in range(len(label[i])):
    #         get_map_labels[i, action.index(label[i][j])] = 1

    # image_route = "/data2/zhangruxu/jingyinuo/action_recognition/dataset/image/"

    # Using CLIP to find the prediction results
    # get_map_preds = torch.zeros((len(image), len(action)))
    for idx in range(len(image)):
        # dirpath = image_route + image[idx]
        image_input = preprocess(Image.open(dirpath)).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"{c}") for c in action]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(len(action))

        # Re-arrange the prediction tensor
        new_values = torch.zeros(len(action))
        new_indices = indices.tolist()
        for i in range(len(action)):
            new_values[i] = values[new_indices.index(i)]
        # get_map_preds[idx,:] = new_values

        # Print the result
        for value, index in zip(values, indices):
            print(f"{action[index]:>16s}: {100 * value.item():.2f}%")

    # map = get_map(get_map_preds.numpy(), get_map_labels.numpy())
    # print(map)