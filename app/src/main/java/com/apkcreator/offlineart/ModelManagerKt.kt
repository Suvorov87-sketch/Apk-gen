package com.apkcreator.offlineart

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.StatFs
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import org.json.JSONObject

class ModelManagerKt(private val ctx: Context) {
    data class MF(val name: String, val url: String, val size: Long)

    fun modelsDir(): File {
        val d = File(ctx.getExternalFilesDir(null), "models")
        d.mkdirs()
        return d
    }

    fun plan(): List<MF> {
        val js = JSONObject(ctx.assets.open("models.json").bufferedReader().readText())
        val arr = js.getJSONArray("files")
        val out = ArrayList<MF>()
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            out.add(MF(o.getString("name"), o.getString("url"), o.getLong("size")))
        }
        return out
    }

    fun missing(): List<MF> {
        val dir = modelsDir()
        return plan().filter { !File(dir, it.name).exists() || File(dir, it.name).length() < it.size }
    }

    fun totalGb(): String {
        val g = plan().sumOf { it.size } / 1073741824.0
        return String.format("%.1f ГБ", g)
    }

    private fun wifiOk(): Boolean {
        val cm = ctx.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val cap = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        return cap.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) || cap.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)
    }

    fun downloadAll(wifiOnly: Boolean, onProg: (Long, Long, String) -> Unit, cancelled: () -> Boolean) {
        if (wifiOnly && !wifiOk()) throw IllegalStateException("Нет Wi-Fi. Отключите опцию Wi-Fi-only или подключитесь к Wi-Fi.")
        val dir = modelsDir()
        val all = plan()
        val total = all.sumOf { it.size }
        var done0 = 0L
        for (f in all) { val lf = File(dir, f.name); if (lf.exists()) done0 += lf.length() }
        val free = StatFs(dir.absolutePath).availableBytes
        if (free < total - done0 + 500L * 1048576L) throw IllegalStateException("Мало места: нужно еще " + ((total - done0) / 1048576) + " МБ")
        var done = done0
        onProg(done, total, "")
        for (f in all) {
            if (cancelled()) throw IllegalStateException("Отменено")
            val lf = File(dir, f.name)
            lf.parentFile?.mkdirs()
            var have = if (lf.exists()) lf.length() else 0L
            if (have >= f.size) { continue }
            val c = URL(f.url).openConnection() as HttpURLConnection
            c.connectTimeout = 30000; c.readTimeout = 60000
            if (have > 0) c.setRequestProperty("Range", "bytes=" + have + "-")
            c.connect()
            val code = c.responseCode
            if (have > 0 && code != 206) { have = 0; lf.delete() }
            else if (have == 0L && code != 200) throw IllegalStateException("HTTP " + code + " для " + f.name)
            RandomAccessFile(lf, "rw").use { raf ->
                raf.seek(have)
                val buf = ByteArray(262144)
                var n = c.inputStream.read(buf)
                var last = 0L
                while (n > 0) {
                    if (cancelled()) throw IllegalStateException("Отменено")
                    raf.write(buf, 0, n)
                    have += n; done += n
                    if (done - last > 1048576) { last = done; onProg(done, total, f.name) }
                    n = c.inputStream.read(buf)
                }
                onProg(done, total, f.name)
            }
            c.disconnect()
            if (lf.length() < f.size) throw IllegalStateException("Файл оборвался: " + f.name)
        }
    }
}