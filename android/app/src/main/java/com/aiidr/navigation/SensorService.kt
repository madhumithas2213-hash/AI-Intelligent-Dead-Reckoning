package com.aiidr.navigation

import android.app.*
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

/**
 * Android Foreground Service acquiring 100Hz Accelerometer, Gyroscope, Magnetometer, and 1Hz GNSS
 */
class SensorService : Service(), SensorEventListener {

    private val binder = LocalBinder()
    private lateinit var sensorManager: SensorManager

    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    private var magSensor: Sensor? = null

    // Latest sensor samples
    var lastAccel = FloatArray(3)
    var lastGyro = FloatArray(3)
    var lastMag = FloatArray(3)

    var listener: SensorDataListener? = null

    interface SensorDataListener {
        fun onSensorFrame(accel: FloatArray, gyro: FloatArray, mag: FloatArray)
    }

    inner class LocalBinder : Binder() {
        fun getService(): SensorService = this@SensorService
    }

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETOMETER)

        startForegroundServiceNotification()
        registerSensors()
    }

    private fun registerSensors() {
        // High frequency sampling (100Hz = 10,000 microseconds)
        accelSensor?.let { sensorManager.registerListener(this, it, 10000) }
        gyroSensor?.let { sensorManager.registerListener(this, it, 10000) }
        magSensor?.let { sensorManager.registerListener(this, it, 10000) }
    }

    private fun startForegroundServiceNotification() {
        val channelId = "ai_idr_service_channel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "AI-IDR Sensor Acquisition Engine",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }

        val notification: Notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("AI-IDR Navigation Active")
            .setContentText("Acquiring 100Hz IMU & GNSS fusion stream...")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

        startForeground(101, notification)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> System.arraycopy(event.values, 0, lastAccel, 0, 3)
            Sensor.TYPE_GYROSCOPE -> System.arraycopy(event.values, 0, lastGyro, 0, 3)
            Sensor.TYPE_MAGNETOMETER -> System.arraycopy(event.values, 0, lastMag, 0, 3)
        }
        listener?.onSensorFrame(lastAccel, lastGyro, lastMag)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
    }
}
