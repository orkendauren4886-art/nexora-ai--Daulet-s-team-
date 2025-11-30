BOT_TOKEN = "8140893616:AAHWFDGPp2Tx7gt43oO3tl4-YpzhbukOVSQ"

# Создаем приложение бота
application = Application.builder().token(BOT_TOKEN).build()

# Пример словаря эмоций животных для демонстрации
emotions_dict = {
    0: "Спокойное",
    1: "Агрессивное",
    2: "Испуганное",
    3: "Игровое",
    4: "Раздраженное",
}

def recognize_animal_and_emotion(audio_bytes):
    speech, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    inputs = feature_extractor(speech, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_id = torch.argmax(logits, dim=-1).item()
        confidence = probs.max().item()
    label = model.config.id2label[predicted_id]

    # Заглушка для эмоций: случайная эмоция для примера
    emotion_id = np.random.choice(list(emotions_dict.keys()))
    emotion_label = emotions_dict.get(emotion_id, "Неизвестно")

    top3 = torch.topk(probs, 3)
    result = f"🐘 {label.upper()} 🎯 Уверенность: {confidence:.1%}\n\n📋 ТОП-3:\n"
    for i, (prob, idx) in enumerate(zip(top3.values[0], top3.indices[0])):
        result += f"{i+1}. {model.config.id2label[idx.item()]}: {prob:.1%}\n"

    result += f"\n💬 Эмоция: {emotion_label}"
    return result

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🐱 Отправь голосовое или wav-файл, я определю животное и эмоцию!", parse_mode="Markdown")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Анализирую звук...")
    voice = await update.voice.get_file()
    audio_bytes = await voice.download_as_bytearray()
    result = recognize_animal_and_emotion(audio_bytes)
    await update.message.reply_text(result, parse_mode="Markdown")

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Анализирую звук...")
    audio = await update.message.audio.get_file()
    audio_bytes = await audio.download_as_bytearray()
    result = recognize_animal_and_emotion(audio_bytes)
    await update.message.reply_text(result, parse_mode="Markdown")

application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.VOICE, handle_voice))
application.add_handler(MessageHandler(filters.AUDIO, handle_audio))

async def run():
    await application.initialize()
    await application.start()
    print("🚀 Бот запущен и готов к работе.")
    await asyncio.Event().wait()

asyncio.get_event_loop().run_until_complete(run())        #бот называется Dauren_bot
