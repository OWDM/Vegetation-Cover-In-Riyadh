if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert('RGB')
    img_array = np.array(image)
    
    # Normalize the image data to 0-1 range
    img_array = img_array / 255.0

    # Resize the image to match model's expected input if necessary
    if img_array.shape != (256, 256, 3):
        img_array = resize(img_array, (256, 256), anti_aliasing=True)

    img_array = np.expand_dims(img_array, axis=0)
    print(img_array.shape)

    placeholderL.image(image)  # This shows the resized and normalized image
    predicted_mask = model.predict(img_array)
    predicted_mask = np.argmax(predicted_mask, axis=-1)
    predicted_mask = predicted_mask[0]  # Remove batch dimension

    # Convert to display format
    predicted_mask = (predicted_mask * 255).astype(np.uint8)
    placeholderR.image(predicted_mask)