/*
  SIT225 Assessment 4 (4P) - Occupancy capture node
  Nguyen Tuan Khoi Le (s226531772)

  Streams one sample per second over Serial for the PIR + ultrasonic fusion
  protocol described in Assessment 3 (Part D.3).

  Wiring (Arduino Nano 33 IoT, 3.3 V logic):
    HC-SR501 PIR   VCC -> external 5 V supply   (GND of that supply tied to Arduino GND)
                   OUT -> D2                    (HC-SR501 OUT swings 0/3.3 V, safe)
    HC-SR04        VCC -> 3V3                   (running at 3.3 V, so Echo is 3.3 V, no divider)
                   TRIG -> D3
                   ECHO -> D4
    DHT22          VCC -> 3V3, DATA -> D5, 10k pull-up between DATA and 3V3

  Ground truth (sitting / away) is marked from the Python UI, not from a
  hardware button, so no button pin is read here.

  Output format, one line per second:
    D,<seq>,<ms>,<pir>,<pir_latch>,<dist_cm>,<us_retry>,<temp_c>,<hum_pct>,<us_err>,<dht_err>
  Fields that could not be read are printed as NaN and flagged in us_err / dht_err.
*/

#include <DHT.h>

#define PIR_PIN    2
#define TRIG_PIN   3
#define ECHO_PIN   4
#define DHT_PIN    5
#define DHT_TYPE   DHT22

// 1 Hz sampling, as specified in the Assessment 3 protocol.
const unsigned long SAMPLE_INTERVAL_MS = 1000;
// DHT22 cannot be polled faster than once every 2 s.
const unsigned long DHT_INTERVAL_MS    = 2000;

// Echo timeout. 25 ms ~ 4.3 m, comfortably past the module's useful range.
const unsigned long ECHO_TIMEOUT_US = 25000UL;
// At 3.3 V the HC-SR04 drops pings more often, so allow a few attempts.
const uint8_t MAX_US_ATTEMPTS = 3;
// The datasheet asks for at least 60 ms between measurement cycles. Retrying
// sooner lets the tail of the previous echo land in the next listening window,
// which showed up as impossible readings (2-5 cm against a 1.4 m target).
const unsigned long US_RETRY_GAP_MS = 60;
// Physical plausibility window for the module (cm).
const float US_MIN_CM = 2.0;
const float US_MAX_CM = 400.0;

DHT dht(DHT_PIN, DHT_TYPE);

unsigned long seq = 0;
unsigned long lastSample = 0;
unsigned long lastDhtRead = 0;

// PIR pulses can be shorter than the sample period, so latch any HIGH seen
// between samples instead of relying on the instantaneous read alone.
bool pirLatch = false;

float lastTemp = NAN;
float lastHum  = NAN;
bool  dhtErr   = true;   // true until the first successful read

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  pinMode(PIR_PIN, INPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  dht.begin();

  // HC-SR501 needs roughly a minute to settle after power-up. Announce it so
  // the logger can warn the operator rather than silently recording noise.
  Serial.println("#SIT225-4P occupancy capture node");
  Serial.println("#columns:seq,ms,pir,pir_latch,dist_cm,us_retry,temp_c,hum_pct,us_err,dht_err");
  Serial.println("#note:PIR needs ~60 s warm-up after power-up");
  Serial.println("READY");

  lastSample = millis();
}

// Single ultrasonic ping. Returns distance in cm, or NAN on timeout / out of range.
float pingOnce()
{
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(3);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    unsigned long dur = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
    if (dur == 0)
        return NAN; // no echo came back in time

    float cm = dur / 58.0; // speed of sound, round trip
    if (cm < US_MIN_CM || cm > US_MAX_CM)
        return NAN;
    return cm;
}

void loop()
{
    // Latch the PIR continuously so a short pulse is not missed between samples.
    if (digitalRead(PIR_PIN) == HIGH)
        pirLatch = true;

    unsigned long now = millis();

    if (now - lastDhtRead >= DHT_INTERVAL_MS)
    {
        lastDhtRead = now;
        float h = dht.readHumidity();
        float t = dht.readTemperature();
        if (isnan(h) || isnan(t))
        {
            dhtErr = true; // keep the previous value, flag it
        }
        else
        {
            lastHum = h;
            lastTemp = t;
            dhtErr = false;
        }
    }

    if (now - lastSample < SAMPLE_INTERVAL_MS)
        return;
    lastSample += SAMPLE_INTERVAL_MS; // fixed cadence, no drift accumulation

    int pirNow = digitalRead(PIR_PIN);

    float dist = NAN;
    uint8_t attempts = 0;
    while (attempts < MAX_US_ATTEMPTS)
    {
        attempts++;
        dist = pingOnce();
        if (!isnan(dist))
            break;
        delay(US_RETRY_GAP_MS); // let the previous echo die away
    }
    bool usErr = isnan(dist);

    seq++;
    Serial.print("D,");
    Serial.print(seq);
    Serial.print(',');
    Serial.print(now);
    Serial.print(',');
    Serial.print(pirNow);
    Serial.print(',');
    Serial.print(pirLatch ? 1 : 0);
    Serial.print(',');
    if (usErr)
        Serial.print("NaN");
    else
        Serial.print(dist, 2);
    Serial.print(',');
    Serial.print(attempts);
    Serial.print(',');
    if (isnan(lastTemp))
        Serial.print("NaN");
    else
        Serial.print(lastTemp, 1);
    Serial.print(',');
    if (isnan(lastHum))
        Serial.print("NaN");
    else
        Serial.print(lastHum, 1);
    Serial.print(',');
    Serial.print(usErr ? 1 : 0);
    Serial.print(',');
    Serial.println(dhtErr ? 1 : 0);

    pirLatch = false; // reset the latch for the next window
}