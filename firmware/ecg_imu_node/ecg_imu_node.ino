/*
 * ESP32 ECG DATA COLLECTOR - FIXED FOR ML MODEL
 *
 * CHANGES FROM ORIGINAL:
 * - Samples ECG at 360 Hz (every 2.78ms)
 * - Sends raw ECG samples on separate MQTT topic
 * - Still shows dashboard data at 1 Hz
 *
 * MQTT TOPICS:
 * - test/sensors : Dashboard data (1 Hz) - EXISTING
 * - test/ecg_raw : Raw ECG for ML model (360 Hz) - NEW
 */

#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// -------- WiFi credentials --------
const char *ssid = "YOUR_WIFI_SSID";
const char *password = "YOUR_WIFI_PASSWORD";

// -------- MQTT broker --------
const char *mqtt_server = "192.168.1.100"; // IP of your MQTT broker
const int mqtt_port = 1883;
const char *mqtt_client_id = "ESP32_TestClient";

WiFiClient espClient;
PubSubClient client(espClient);

// ==================== MPU6050 CONFIGURATION ====================
const int MPU_ADDR = 0x68;
int16_t accelerometer_x, accelerometer_y, accelerometer_z;
int16_t gyroscope_x, gyroscope_y, gyroscope_z;
float accel_x, accel_y, accel_z;
float gyro_x, gyro_y, gyro_z;
float totalAcceleration = 0.0;
float totalGyro = 0.0;

// Fall detection variables (unchanged)
#define FALL_THRESHOLD 3.5
#define IMPACT_DURATION 500
#define STILLNESS_THRESHOLD 0.5
#define STILLNESS_DURATION 4000
#define DEBOUNCE_TIME 10000
#define GYRO_THRESHOLD 400
#define POST_FALL_GYRO_THRESH 20
#define FALL_CONFIRM_TIMEOUT 5000
#define MOVEMENT_CANCEL_THRESH 2.0
#define FALL_ALERT_DURATION 60000

unsigned long lastFallTime = 0;
bool potentialFall = false;
unsigned long fallDetectedTime = 0;
bool stillnessDetected = false;
unsigned long stillnessTime = 0;
unsigned long fallConfirmedTime = 0;

// ==================== AD8232 CONFIGURATION ====================
const int ECG_OUTPUT_PIN = 34;
const int LO_PLUS_PIN = 32;
const int LO_MINUS_PIN = 33;

// ECG Signal Processing Constants
#define ADC_RESOLUTION 4095.0 // 12-bit ADC
#define ADC_VOLTAGE_REF 3.3   // ESP32 reference voltage
#define AD8232_GAIN 1000.0    // AD8232 typical gain (1000-1100)
#define BASELINE_SAMPLES 100  // Samples for baseline calculation
#define SCALE_FACTOR 1.0      // Scaling to match MIT-BIH amplitude

// Baseline tracking for DC offset removal
float ecgBaseline = 2047.5;   // Start at mid-range
float baselineAlpha = 0.0001; // Very slow baseline tracking (was 0.001)
bool baselineInitialized = false;
int baselineCounter = 0;

// Simple moving average filter for noise reduction
#define FILTER_SIZE 7 // Increased from 3 to 7 for stronger smoothing
int filterBuffer[FILTER_SIZE] = {0};
int filterIndex = 0;

// ==================== TIMING VARIABLES ====================
// NEW: Separate timing for ECG sampling and dashboard updates
unsigned long lastECGSampleTime = 0;
unsigned long lastDashboardTime = 0;
unsigned long lastIMUSampleTime = 0;

const unsigned long ECG_SAMPLE_INTERVAL = 2778; // 360 Hz = 1/360 = 0.00278s = 2778 microseconds
const unsigned long DASHBOARD_INTERVAL = 1000;  // 1 Hz for dashboard
const unsigned long IMU_SAMPLE_INTERVAL = 500;  // 2 Hz IMU updates (heavily reduced to avoid blocking ECG)

// ECG buffer for batch sending - OPTIMIZED for 360 Hz
#define ECG_BUFFER_SIZE 360 // 1 full second at 360 Hz
int ecgBuffer[ECG_BUFFER_SIZE];
int bufferIndex = 0;

// Preallocated message buffer for faster transmission
char msgBuffer[ECG_BUFFER_SIZE * 5]; // ~5 chars per number max

// DIAGNOSTIC: Track MQTT publish success
unsigned long publishCount = 0;
unsigned long publishFailCount = 0;
unsigned long lastDiagnosticTime = 0;

void setup_wifi()
{
  delay(10);
  Serial.begin(115200);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void reconnect()
{
  while (!client.connected())
  {
    Serial.print("Connecting to MQTT...");
    if (client.connect(mqtt_client_id))
    {
      Serial.println("connected!");
    }
    else
    {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
}

void setup()
{
  Serial.begin(115200);

  // Initialize MPU6050
  Wire.begin(4, 5);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  // Initialize AD8232
  pinMode(LO_PLUS_PIN, INPUT);
  pinMode(LO_MINUS_PIN, INPUT);

  Serial.println("==========================================");
  Serial.println("ECG + Fall Detection System (360 Hz ECG)");
  Serial.println("==========================================");

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setKeepAlive(60);

  // OPTIMIZED WiFi and MQTT settings for high throughput
  WiFi.setSleep(false);                // Disable power saving
  WiFi.setTxPower(WIFI_POWER_19_5dBm); // Maximum WiFi power

  // Increase MQTT buffer size for larger batches (360 samples)
  client.setBufferSize(2048); // Large enough for 360 samples

  delay(1000);
}

void loop()
{
  if (!client.connected())
    reconnect();
  client.loop();

  unsigned long currentTime = millis();
  unsigned long currentMicros = micros();

  // ==================== HIGH-SPEED ECG SAMPLING (360 Hz) ====================
  if (currentMicros - lastECGSampleTime >= ECG_SAMPLE_INTERVAL)
  {
    lastECGSampleTime = currentMicros;

    // Read raw ECG value
    int rawECGValue = 0;
    bool leadsOff = false;

    if (digitalRead(LO_PLUS_PIN) == HIGH || digitalRead(LO_MINUS_PIN) == HIGH)
    {
      rawECGValue = 0;
      leadsOff = true;
    }
    else
    {
      rawECGValue = analogRead(ECG_OUTPUT_PIN);
    }

    // ==================== CONVERT TO MILLIVOLTS (MIT-BIH format) ====================
    int processedValue = 0; // Default to 0 if leads off

    if (!leadsOff)
    {
      // Initialize baseline on first samples
      if (!baselineInitialized)
      {
        if (baselineCounter < BASELINE_SAMPLES)
        {
          ecgBaseline = (ecgBaseline * baselineCounter + rawECGValue) / (baselineCounter + 1);
          baselineCounter++;
          processedValue = 0; // Don't process during initialization
        }
        else
        {
          baselineInitialized = true;
        }
      }

      if (baselineInitialized)
      {
        // Update baseline using exponential moving average (high-pass filter effect)
        ecgBaseline = (baselineAlpha * rawECGValue) + ((1.0 - baselineAlpha) * ecgBaseline);

        // Remove DC offset (subtract baseline)
        float acCoupled = rawECGValue - ecgBaseline;

        // Convert ADC to voltage: V = (ADC / 4095) * 3.3V
        float voltage = (acCoupled / ADC_RESOLUTION) * ADC_VOLTAGE_REF;

        // Convert to millivolts and account for AD8232 gain
        // AD8232 gain is typically 1000-1100 (depends on circuit)
        // Divide by gain to get original physiological signal
        float millivolts = (voltage * 1000.0) / AD8232_GAIN;

        // Apply additional scaling to match MIT-BIH amplitude if needed
        millivolts = millivolts * SCALE_FACTOR;

        // Store as integer with 3 decimal precision (multiply by 1000)
        // This stores values like: 1250 = 1.25mV, -500 = -0.5mV
        processedValue = (int)(millivolts * 1000.0);

        // Clamp to reasonable ECG range (±5mV to prevent extreme outliers)
        if (processedValue > 5000)
          processedValue = 5000;
        if (processedValue < -5000)
          processedValue = -5000;

        // Apply simple moving average filter to reduce noise
        filterBuffer[filterIndex] = processedValue;
        filterIndex = (filterIndex + 1) % FILTER_SIZE;

        int sum = 0;
        for (int i = 0; i < FILTER_SIZE; i++)
        {
          sum += filterBuffer[i];
        }
        processedValue = sum / FILTER_SIZE;
      }
    }

    // Buffer the processed samples
    ecgBuffer[bufferIndex++] = processedValue;

    // Send batch when buffer is full (every 1 second now)
    // FAILSAFE: Publish at 355+ samples to avoid getting stuck at 356
    if (bufferIndex >= 355)
    {
      // Pad with last value if needed to reach 360
      while (bufferIndex < 360)
      {
        ecgBuffer[bufferIndex] = ecgBuffer[bufferIndex - 1];
        bufferIndex++;
      }

      int pos = 0;
      for (int i = 0; i < 360; i++)
      {
        pos += sprintf(msgBuffer + pos, "%d", ecgBuffer[i]);
        if (i < 359)
          msgBuffer[pos++] = ',';
      }
      msgBuffer[pos] = '\0';

      // DIAGNOSTIC: Track publish success
      bool published = client.publish("test/ecg_raw", msgBuffer, false);
      if (published)
      {
        publishCount++;
      }
      else
      {
        publishFailCount++;
        Serial.println("⚠️  MQTT PUBLISH FAILED!");
      }

      bufferIndex = 0;
    }
  }

  // TEMPORARILY DISABLED IMU FOR ECG TESTING
  // Uncomment this block once ECG is working at 360Hz
  /*
  if (currentTime - lastIMUSampleTime >= IMU_SAMPLE_INTERVAL)
  {
    // Skip IMU if ECG sample is due within 1ms to avoid blocking
    unsigned long timeSinceLastECG = (currentMicros - lastECGSampleTime);
    if (timeSinceLastECG > 1000) {  // Only read IMU if we're NOT close to ECG sample time
      lastIMUSampleTime = currentTime;

      // ==================== READ MPU6050 (for fall detection) ====================
      Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 14, true);

    accelerometer_x = Wire.read() << 8 | Wire.read();
    accelerometer_y = Wire.read() << 8 | Wire.read();
    accelerometer_z = Wire.read() << 8 | Wire.read();
    Wire.read();
    Wire.read(); // Skip temperature
    gyroscope_x = Wire.read() << 8 | Wire.read();
    gyroscope_y = Wire.read() << 8 | Wire.read();
    gyroscope_z = Wire.read() << 8 | Wire.read();

    accel_x = (accelerometer_x / 16384.0) * 9.81;
    accel_y = (accelerometer_y / 16384.0) * 9.81;
    accel_z = (accelerometer_z / 16384.0) * 9.81;

    gyro_x = gyroscope_x / 131.0;
    gyro_y = gyroscope_y / 131.0;
    gyro_z = gyroscope_z / 131.0;

    // ==================== FALL DETECTION (unchanged) ====================
    float accel_x_g = accel_x / 9.81;
    float accel_y_g = accel_y / 9.81;
    float accel_z_g = accel_z / 9.81;
    totalAcceleration = sqrt(accel_x_g * accel_x_g + accel_y_g * accel_y_g + accel_z_g * accel_z_g);
    totalGyro = sqrt(gyro_x * gyro_x + gyro_y * gyro_y + gyro_z * gyro_z);

    if (!potentialFall && !stillnessDetected &&
        (totalAcceleration > FALL_THRESHOLD || totalGyro > GYRO_THRESHOLD) &&
        (currentTime - lastFallTime > DEBOUNCE_TIME))
    {
      potentialFall = true;
      fallDetectedTime = currentTime;
      Serial.println("\n----- POTENTIAL FALL DETECTED! -----");
    }

    if (potentialFall && !stillnessDetected &&
        (currentTime - fallDetectedTime > IMPACT_DURATION))
    {
      if (abs(totalAcceleration - 1.0) < STILLNESS_THRESHOLD &&
          totalGyro < POST_FALL_GYRO_THRESH)
      {
        stillnessDetected = true;
        stillnessTime = currentTime;
        Serial.println("\n----- POST-FALL STILLNESS DETECTED -----");
      }
      if (currentTime - fallDetectedTime > FALL_CONFIRM_TIMEOUT)
      {
        potentialFall = false;
        stillnessDetected = false;
      }
    }

    if (stillnessDetected)
    {
      if (totalAcceleration > MOVEMENT_CANCEL_THRESH || totalGyro > POST_FALL_GYRO_THRESH * 2)
      {
        stillnessDetected = false;
        potentialFall = false;
        Serial.println("\n----- FALL ALERT CANCELED -----");
      }
      else if (currentTime - stillnessTime > STILLNESS_DURATION)
      {
        triggerAlert();
        lastFallTime = currentTime;
        fallConfirmedTime = currentTime;
        potentialFall = false;
        stillnessDetected = false;
      }
    }
    } // Close the IMU skip check
  }
  */
  // End of IMU disabled block

  // ==================== DASHBOARD UPDATE (1 Hz) ====================
  if (currentTime - lastDashboardTime >= DASHBOARD_INTERVAL)
  {
    lastDashboardTime = currentTime;

    // Get current ECG value for dashboard display (show the last processed value)
    int currentECGValue = (bufferIndex > 0) ? ecgBuffer[bufferIndex - 1] : 0;

    int fallStatus = (fallConfirmedTime > 0 && currentTime - fallConfirmedTime <= FALL_ALERT_DURATION) ? 1 : 0;

    // Send dashboard data (1 Hz - same as before)
    char msg[128];
    sprintf(msg, "accel accel_x=%f,accel_y=%f,accel_z=%f,fallStatus=%d,ecgValue=%d",
            accel_x, accel_y, accel_z, fallStatus, currentECGValue);
    client.publish("test/sensors", msg);

    // Print status to serial
    Serial.print("Dashboard update | ECG: ");
    Serial.print(currentECGValue);
    Serial.print(" | Fall: ");
    Serial.print(fallStatus);
    Serial.print(" | Accel: ");
    Serial.println(totalAcceleration);

    // DIAGNOSTIC: Print MQTT stats every 10 seconds
    if (currentTime - lastDiagnosticTime >= 10000)
    {
      lastDiagnosticTime = currentTime;

      // Calculate actual sampling rate
      float actualRate = (publishCount * 360.0 + bufferIndex) / 10.0;

      Serial.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
      Serial.println("📊 MQTT DIAGNOSTICS (last 10 sec):");
      Serial.print("   Publishes: ");
      Serial.print(publishCount);
      Serial.print(" (should be ~10)");
      if (publishCount < 8)
      {
        Serial.print(" ⚠️  LOW!");
      }
      Serial.println();
      Serial.print("   Failures: ");
      Serial.println(publishFailCount);
      Serial.print("   MQTT Connected: ");
      Serial.println(client.connected() ? "YES ✓" : "NO ❌");
      Serial.print("   Buffer Index: ");
      Serial.print(bufferIndex);
      Serial.println(" (should reset to 0 after publish)");
      Serial.print("   Actual Rate: ");
      Serial.print(actualRate);
      Serial.print(" Hz (target: 360 Hz)");
      if (actualRate < 300)
      {
        Serial.print(" ⚠️  TOO LOW!");
      }
      Serial.println();
      Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

      // Reset counters for next interval
      publishCount = 0;
      publishFailCount = 0;
    }
  }

  // No delay - we need maximum speed for 360 Hz sampling
  // The timing is controlled by ECG_SAMPLE_INTERVAL
}

void triggerAlert()
{
  Serial.println("\n\n!!!!!! FALL CONFIRMED - SENDING ALERT !!!!!!");
  Serial.println("Alert completed.\n\n");
}