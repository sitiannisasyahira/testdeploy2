elif page == "Klasifikasi":
    st.header("🌿 Klasifikasi Daun (Sehat / Tidak Sehat)")
    st.write("Unggah gambar daun. Jika hasilnya keluar, tunggu efek *alay spektakuler!* 🌈")

    uploaded = st.file_uploader("Unggah gambar daun", type=["jpg","jpeg","png"])
    if uploaded is None:
        st.info("Silakan unggah gambar.")
    else:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Input gambar", use_container_width=True)

            arr = preprocess_for_classifier(img, classifier)
            with st.spinner("Menjalankan klasifikasi... 🌪️"):
                pred = classifier.predict(arr)

            cls_id = int(np.argmax(pred))
            conf = float(np.max(pred))
            label = "🌱 Daun Sehat" if cls_id == 0 else "🍂 Daun Tidak Sehat"

            # 🎉 Efek alay dimulai
            st.balloons()

            # CSS animasi
            st.markdown("""
                <style>
                @keyframes rainbow {
                    0% {color: red;}
                    20% {color: orange;}
                    40% {color: yellow;}
                    60% {color: green;}
                    80% {color: blue;}
                    100% {color: purple;}
                }
                @keyframes shake {
                    0% { transform: translate(1px, 1px) rotate(0deg); }
                    10% { transform: translate(-1px, -2px) rotate(-1deg); }
                    20% { transform: translate(-3px, 0px) rotate(1deg); }
                    30% { transform: translate(3px, 2px) rotate(0deg); }
                    40% { transform: translate(1px, -1px) rotate(1deg); }
                    50% { transform: translate(-1px, 2px) rotate(-1deg); }
                    60% { transform: translate(-3px, 1px) rotate(0deg); }
                    70% { transform: translate(3px, 1px) rotate(-1deg); }
                    80% { transform: translate(-1px, -1px) rotate(1deg); }
                    90% { transform: translate(1px, 2px) rotate(0deg); }
                    100% { transform: translate(1px, -2px) rotate(-1deg); }
                }
                .alay-text {
                    font-size: 2rem;
                    font-weight: 900;
                    text-align: center;
                    animation: rainbow 2s infinite linear, shake 0.3s infinite;
                    text-shadow: 2px 2px 10px #ff00ff;
                }
                .confetti {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                    z-index: 9999;
                    background: repeating-linear-gradient(45deg, 
                        rgba(255,0,0,0.1), rgba(255,0,0,0.1) 10px, 
                        rgba(0,255,0,0.1) 10px, rgba(0,255,0,0.1) 20px,
                        rgba(0,0,255,0.1) 20px, rgba(0,0,255,0.1) 30px);
                    animation: fade 3s ease-out;
                }
                @keyframes fade {
                    from {opacity: 1;}
                    to {opacity: 0;}
                }
                </style>
                <div class="confetti"></div>
                <div class="alay-text">💥 HASIL: %s 💥<br>🔥 Confidence: %.2f 🔥</div>
            """ % (label, conf), unsafe_allow_html=True)

        except Exception as e:
            st.error("Terjadi kesalahan saat klasifikasi:")
            st.exception(e)
