package com.apkcreator.offlineart

import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.util.concurrent.Executors

class MainActivityKt : AppCompatActivity() {
    private var pipe: DiffusionPipelineKt? = null
    private val bg = Executors.newSingleThreadExecutor()
    private lateinit var promptInput: EditText
    private lateinit var negInput: EditText
    private lateinit var seedInput: EditText
    private lateinit var goBtn: Button
    private lateinit var saveBtn: Button
    private lateinit var statusText: TextView
    private lateinit var bar: ProgressBar
    private lateinit var img: ImageView
    private lateinit var styleRow: LinearLayout
    private var styleSuffix = ""
    private var lastBitmap: Bitmap? = null
    private var generating = false

    private val styles = listOf(
        Pair("Фото", ", ultra detailed photograph, 85mm, natural light, high detail"),
        Pair("Аниме", ", anime style, vibrant, detailed anime key visual, studio quality"),
        Pair("Фэнтези", ", epic fantasy digital painting, intricate details, dramatic lighting"),
        Pair("Киберпанк", ", cyberpunk, neon lights, futuristic city, cinematic lighting"),
        Pair("Живопись", ", oil painting, textured brushstrokes, gallery masterpiece")
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "NeuroMax — офлайн нейро-арт"
        val root = ScrollView(this)
        val col = LinearLayout(this)
        col.orientation = LinearLayout.VERTICAL
        val pad = (16 * resources.displayMetrics.density).toInt()
        col.setPadding(pad, pad, pad, pad)
        root.addView(col)
        setContentView(root)

        col.addView(mkLabel("Промпт (что нарисовать):"))
        promptInput = EditText(this)
        promptInput.hint = "Например: fox in neon forest, sunset"
        promptInput.minLines = 3
        col.addView(promptInput)
        col.addView(mkLabel("Стиль (все открыты, без подписок):"))
        styleRow = LinearLayout(this)
        styleRow.orientation = LinearLayout.HORIZONTAL
        col.addView(styleRow)
        styles.forEachIndexed { i, s ->
            val b = Button(this, null, android.R.attr.buttonStyleSmall)
            b.text = s.first
            b.setOnClickListener { selectStyle(i) }
            b.tag = i
            styleRow.addView(b)
        }
        col.addView(mkLabel("Негативный промпт (работает при CFG > 1):"))
        negInput = EditText(this)
        negInput.setText("blurry, low quality, watermark, distorted, deformed")
        col.addView(negInput)
        val row = LinearLayout(this)
        row.orientation = LinearLayout.HORIZONTAL
        col.addView(row)
        val seedBox = LinearLayout(this)
        seedBox.orientation = LinearLayout.VERTICAL
        seedBox.addView(mkLabel("Сид:"))
        seedInput = EditText(this)
        seedInput.hint = "пусто = случайно"
        seedInput.inputType = android.text.InputType.TYPE_CLASS_NUMBER
        seedBox.addView(seedInput)
        row.addView(seedBox, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        val dice = Button(this)
        dice.text = "Кубик"
        dice.setOnClickListener { seedInput.setText((System.currentTimeMillis() % 999999).toString()) }
        row.addView(dice)
        col.addView(mkLabel("CFG (1.0 = быстрый LCM, выше = точнее и медленнее):"))
        val cfgBar = SeekBar(this)
        cfgBar.max = 65
        cfgBar.progress = 0
        col.addView(cfgBar)
        val cfgText = mkLabel("CFG = 1.0")
        col.addView(cfgText)
        cfgBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, u: Boolean) { cfgText.text = "CFG = " + (1.0 + p / 10.0).toString() }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })
        val wifi = CheckBox(this)
        wifi.text = "Качать модель только по Wi-Fi"
        wifi.isChecked = true
        col.addView(wifi)
        goBtn = Button(this)
        goBtn.text = "Сгенерировать (4 шага LCM)"
        col.addView(goBtn)
        bar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal)
        bar.visibility = View.GONE
        col.addView(bar)
        statusText = TextView(this)
        statusText.text = "При первом запуске скачивается модель (~4.1 ГБ fp32, нужен Wi-Fi и место). Дальше всё офлайн."
        col.addView(statusText)
        img = ImageView(this)
        img.adjustViewBounds = true
        col.addView(img)
        saveBtn = Button(this)
        saveBtn.text = "Сохранить в галерею"
        saveBtn.isEnabled = false
        col.addView(saveBtn)
        selectStyle(1)
        goBtn.setOnClickListener {
            if (generating) return@setOnClickListener
            val cfg = 1.0 + cfgBar.progress / 10.0
            startGenerate(wifi.isChecked, cfg.toFloat())
        }
        saveBtn.setOnClickListener { lastBitmap?.let { b -> saveToGallery(b) } }
    }

    private fun mkLabel(t: String): TextView {
        val v = TextView(this)
        v.text = t
        v.setPadding(0, 16, 0, 8)
        return v
    }

    private fun selectStyle(i: Int) {
        styleSuffix = styles[i].second
        for (k in 0 until styleRow.childCount) {
            val b = styleRow.getChildAt(k) as Button
            b.isEnabled = (b.tag as Int) != i
        }
    }

    private fun say(t: String, pct: Int = -1) {
        runOnUiThread {
            statusText.text = t
            if (pct < 0) { bar.visibility = View.GONE } else { bar.visibility = View.VISIBLE; bar.progress = pct }
        }
    }

    private fun startGenerate(wifiOnly: Boolean, cfg: Float) {
        generating = true
        goBtn.isEnabled = false
        saveBtn.isEnabled = false
        val prompt = promptInput.text.toString().trim().ifEmpty { "magical landscape" } + styleSuffix
        val neg = negInput.text.toString().trim()
        val seed = seedInput.text.toString().toLongOrNull() ?: (System.currentTimeMillis() % 1000000)
        bg.execute {
            try {
                val mgr = ModelManagerKt(this)
                val miss = mgr.missing()
                if (miss.isNotEmpty()) {
                    say("Качаю модель: 0% (всего " + mgr.totalGb() + ")")
                    mgr.downloadAll(wifiOnly, { done, total, _ ->
                        say("Качаю модель: " + (done * 100 / total) + "%", (done * 100 / total).toInt())
                    }, { !generating })
                }
                if (pipe == null) {
                    say("Загружаю нейросеть в память...", 100)
                    pipe = DiffusionPipelineKt(mgr.modelsDir(), Runtime.getRuntime().availableProcessors())
                }
                say("Рисую (шаг 0/4)...", 5)
                val bmp = pipe!!.generate(prompt, neg, seed, 4, cfg, 512) { s, t ->
                    say("Рисую (шаг " + s + "/" + t + ")...", 5 + s * 90 / t)
                }
                lastBitmap = bmp
                runOnUiThread { img.setImageBitmap(bmp); saveBtn.isEnabled = true }
                say("Готово! Сид: " + seed)
            } catch (e: Exception) {
                say("Ошибка: " + (e.message ?: e.toString()))
            } finally {
                generating = false
                runOnUiThread { goBtn.isEnabled = true }
            }
        }
    }

    private fun saveToGallery(bmp: Bitmap) {
        try {
            val name = "neuromax_" + System.currentTimeMillis() + ".jpg"
            val cv = android.content.ContentValues()
            cv.put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, name)
            cv.put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            cv.put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, "Pictures/NeuroMax")
            val uri = contentResolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv)
            if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.JPEG, 92, it) }
                statusText.text = "Сохранено: " + name
            }
        } catch (e: Exception) { statusText.text = "Не сохранено: " + (e.message ?: e.toString()) }
    }

    override fun onDestroy() {
        generating = false
        bg.shutdownNow()
        try { pipe?.close() } catch (e: Exception) {}
        super.onDestroy()
    }
}