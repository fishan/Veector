def local_fine_tuning(user_data):
    # Загрузка только необходимых блоков
    classifier = matrix.load_block('classifier', block_hashes['classifier'])
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
    
    for text, label in user_data:
        inputs = tokenizer(text, return_tensors='pt').to(device)
        outputs = dynamic_inference(text)
        loss = F.cross_entropy(outputs, label)
        loss.backward()
        optimizer.step()
    
    # Сохранение обновлений в IPFS
    new_hash = client.add('classifier_updated.pt')['Hash']
    return new_hash

def sync_updates(new_hash):
    # Шифрование обновлений
    encrypted = encrypt(new_hash)
    client.add(encrypted)
    p2p.broadcast(encrypted)