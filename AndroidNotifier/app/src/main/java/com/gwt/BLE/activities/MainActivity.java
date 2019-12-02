package com.gwt.BLE.activities;

import android.app.ActionBar;
import android.app.Activity;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattService;
import android.content.Intent;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.text.TextUtils;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.gwt.BLE.R;
import com.ublox.BLE.interfaces.BluetoothDeviceRepresentation;
import com.ublox.BLE.services.BluetoothLeService;
import com.ublox.BLE.services.BluetoothLeServiceReceiver;
import com.ublox.BLE.utils.ConnectionState;
import com.ublox.BLE.utils.GattAttributes;

import java.util.HashMap;
import java.util.Locale;
import java.util.Random;
import java.util.UUID;

import static com.ublox.BLE.services.BluetoothLeService.ITEM_TYPE_NOTIFICATION;
import static com.ublox.BLE.services.BluetoothLeService.ITEM_TYPE_READ;
import static com.ublox.BLE.services.BluetoothLeService.ITEM_TYPE_WRITE;

public class MainActivity extends Activity {

    private static final String TAG = "MyBleActivity";

    public static final String EXTRA_DEVICE = "device";
    public static final String EXTRA_REMOTE = "remote";

    private static final byte BLE_STRANGER = (byte)0x10;
    private static final byte BLE_USER = (byte)0x20;

    private boolean isRemoteMode;
    private TextView tvStatus;
    private TextView nameView;
    private RelativeLayout rlProgress;

    private BluetoothDeviceRepresentation mDevice;

    private BluetoothLeService mBluetoothLeService;
    private static ConnectionState mConnectionState = ConnectionState.DISCONNECTED;

    private BluetoothGattCharacteristic characteristicFifo;

    private TextToSpeech ttsSpeaker;

    public void onServiceConnected() {
        if (!mBluetoothLeService.initialize(this)) {
            finish();
        }
    }

    public final MyBroadcastReceiver mGattUpdateReceiver = new MyBroadcastReceiver();

    private void updateStatus() {
        switch (mConnectionState) {
            case DISCONNECTED:
                tvStatus.setText(R.string.status_disconnected);
                break;
            case CONNECTING:
                if(!isRemoteMode) {
                    tvStatus.setText(R.string.status_connecting);
                }
                break;
            case CONNECTED:
                tvStatus.setText(R.string.status_connected);
                break;
            case BLE_EXCHANGE:
                tvStatus.setText("loading data");
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (mBluetoothLeService != null) {
            mBluetoothLeService.register(mGattUpdateReceiver);
            final boolean result = mBluetoothLeService.connect(mDevice);
            Log.d(TAG, "Connect request result=" + result);
            mConnectionState = ConnectionState.CONNECTING;
            invalidateOptionsMenu();
            updateStatus();
            rlProgress.setVisibility(View.VISIBLE);
        }
        invalidateOptionsMenu();
    }

    @Override
    protected void onPause() {
        super.onPause();
        try {
            mBluetoothLeService.disconnect();
            mBluetoothLeService.close();
            mConnectionState = ConnectionState.DISCONNECTED;
            mBluetoothLeService.unregister();
        } catch (Exception ignore) {}
        invalidateOptionsMenu();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getActionBar().setTitle("");
        getActionBar().setLogo(R.drawable.logo);
        getActionBar().setDisplayUseLogoEnabled(true);
        setContentView(R.layout.activity_main);

        ttsSpeaker = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status == TextToSpeech.ERROR) {
                    Log.d(TAG, "TTS init failed with error " + status);
                } else {
                    Log.d(TAG, "TTS init successful!");
                    int st = ttsSpeaker.setLanguage(Locale.US);
                    Log.d(TAG, "SetLanguage returns " + st);
//                    st = ttsSpeaker.setOnUtteranceCompletedListener(
//                            new TextToSpeech.OnUtteranceCompletedListener() {
//                                @Override
//                                public void onUtteranceCompleted(String utteranceId) {
//                                    Log.d(TAG, "TTS onUtteranceCompleted happened!");
//                                    mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{0x00});
//                                }
//                            });
//                    Log.d(TAG, "setOnUtteranceCompletedListener returns " + st);
                    st = ttsSpeaker.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                        @Override
                        public void onStart(String utteranceId) {
                            Log.d(TAG, "TTS onStart happened!");
                        }

                        public void onStop (String utteranceId, boolean interrupted) {
                            Log.d(TAG, "TTS onStop happened!");
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{0x00});
                        }

                        @Override
                        public void onDone(String utteranceId) {
                            Log.d(TAG, "TTS onDone happened!");
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{0x00});
                        }

                        @Override
                        public void onError(String utteranceId) {
                            Log.d(TAG, "TTS onError happened!");
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{0x00});
                        }
                    });

                    Log.d(TAG, "setOnUtteranceProgressListener returns " + st);
                }
            }
        });

//        player = new MediaPlayer();
//        player.setOnCompletionListener(mp -> {
//            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{0x00});
//        });

        tvStatus = findViewById(R.id.tvStatus);
        rlProgress = findViewById(R.id.rlProgress);
        nameView = findViewById(R.id.NameView);

        final Intent intent = getIntent();
        isRemoteMode = intent.hasExtra(EXTRA_REMOTE);
        updateStatus();

        if(isRemoteMode) {

            rlProgress.setVisibility(View.GONE);

        } else {
            connectToDevice((BluetoothDeviceRepresentation) intent.getParcelableExtra(EXTRA_DEVICE));

            // Get a ref to the actionbar and set the navigation mode
            final ActionBar actionBar = getActionBar();

            final String name = mDevice.getName();
            if (!TextUtils.isEmpty(name)) {
                getActionBar().setTitle(name);
            } else {
                getActionBar().setTitle(mDevice.getAddress());
            }

            actionBar.setDisplayShowCustomEnabled(true);
        }
    }

    private void connectToDevice(BluetoothDeviceRepresentation bluetoothDevice) {
        // get the information from the device scan
        mDevice = bluetoothDevice;

        mBluetoothLeService = new BluetoothLeService();
        onServiceConnected();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_connected, menu);
        if(!isRemoteMode) {
            switch (mConnectionState) {
                case CONNECTED:
                    Log.d(TAG, "Create menu in Connected mode");
                    menu.findItem(R.id.menu_connect).setVisible(false);
                    menu.findItem(R.id.menu_disconnect).setVisible(true);
                    break;
                case CONNECTING:
                    Log.d(TAG, "Create menu in Connecting mode");
                    menu.findItem(R.id.menu_connect).setVisible(false);
                    menu.findItem(R.id.menu_disconnect).setVisible(false);
                    break;
                case DISCONNECTED:
                    Log.d(TAG, "Create menu in Disconnected mode");
                    menu.findItem(R.id.menu_connect).setVisible(true);
                    menu.findItem(R.id.menu_disconnect).setVisible(false);
                    break;
                case BLE_EXCHANGE:
                    Log.d(TAG, "Create menu in Exchange mode");
                    menu.findItem(R.id.menu_connect).setVisible(false);
                    menu.findItem(R.id.menu_disconnect).setVisible(false);
                    break;
            }
        } else {
            menu.findItem(R.id.menu_connect).setVisible(false);
            menu.findItem(R.id.menu_disconnect).setVisible(false);
            menu.findItem(R.id.menu_refresh).setVisible(false);
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch(item.getItemId()) {
            case R.id.menu_connect:
                mBluetoothLeService.connect(mDevice);
                mConnectionState = ConnectionState.CONNECTING;
                invalidateOptionsMenu();
                updateStatus();
                rlProgress.setVisibility(View.VISIBLE);
                return true;
            case R.id.menu_disconnect:
                mBluetoothLeService.disconnect();
                updateStatus();
                rlProgress.setVisibility(View.VISIBLE);
                return true;
            case android.R.id.home:
                onBackPressed();
                return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private class MyBroadcastReceiver implements BluetoothLeServiceReceiver {
        @Override
        public void onDescriptorWrite() {
            Log.d(TAG, "onDescriptorWrite call");
        }

        @Override
        public void onPhyAvailable(boolean isUpdate) {
            Log.d(TAG, "onPhyAvailable call");
        }

        @Override
        public void onMtuUpdate(int mtu, int status) {
            Log.d(TAG, "onMtuUpdate call");
        }

        @Override
        public void onRssiUpdate(int rssi) {
            Log.d(TAG, "onRssiUpdate call");
        }

       @Override
        public void onDataAvailable(UUID uUid, int type, byte[] data) {
           String typeString = "";
           switch (type) {
               case ITEM_TYPE_READ:
                   typeString = "ITEM_TYPE_READ";
                   break;
               case ITEM_TYPE_WRITE:
                   typeString = "ITEM_TYPE_WRITE";
                   break;
               case ITEM_TYPE_NOTIFICATION:
                   typeString = "ITEM_TYPE_NOTIFICATION";
                   break;
               default:
                   typeString = "UNKNOWN";
           }
           Log.d(TAG, "onDataAvailable call: Data type " + typeString + " with size " + data.length + " is available!");

           // Tracks location: /mnt/sdcard/Documents/Sounds

           if(type == ITEM_TYPE_NOTIFICATION) {
               switch (data[0]) {
                   case BLE_STRANGER:
                       ttsSpeaker.speak("Stop, Stranger!", TextToSpeech.QUEUE_FLUSH, null);
//                       try {
//                           player.reset();
//                           player.setDataSource("/mnt/sdcard/Documents/Sounds/Stranger.mp3");
//                           player.prepare();
//                           player.start();
//                       } catch (IOException e) {
//                           Log.d(TAG, "Playback exception!");
//                           e.printStackTrace();
//                           mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{0x00});
//                       }
                       runOnUiThread(() -> {
                           nameView.setText("Stop, stranger!");
                       });
                       Log.d(TAG, "Stranger detected!");
                       break;
                   case BLE_USER:
                       int len;
                       for (len = 1; len < data.length && data[len] != 0; len++) { }
                       String name = new String(data, 1, len-1);
                       String mostRecentUtteranceID = (new Random().nextInt() % 9999999) + ""; // "" is String force

                       // set params
                       // *** this method will work for more devices: API 19+ ***
                       HashMap<String, String> params = new HashMap<>();
                       params.put(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, mostRecentUtteranceID);

                       ttsSpeaker.speak("Hi, " + name + "!", TextToSpeech.QUEUE_FLUSH, params);
//                       try {
//                           player.reset();
//                           player.setDataSource("/mnt/sdcard/Documents/Sounds/" + name + ".mp3");
//                           player.prepare();
//                           player.start();
//                       } catch (IOException e) {
//                           Log.d(TAG, "Playback exception!");
//                           e.printStackTrace();
//                           mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{0x00});
//                       }
                       Log.d(TAG, "User name " + name);
                       runOnUiThread(() -> {
                           nameView.setText("Hi, " + name);
                       });
                       break;
                   default:
                       Log.d(TAG, "Invalid request  " + data[0]);
               }
           }
        }

        @Override
        public void onServicesDiscovered() {
            runOnUiThread(() -> {
                updateStatus();
                for (BluetoothGattService service : mBluetoothLeService.getSupportedGattServices()) {
                    for (BluetoothGattCharacteristic characteristic : service.getCharacteristics()) {
                        String uuid = characteristic.getUuid().toString();
                        if (uuid.equals(GattAttributes.UUID_CHARACTERISTIC_FIFO)) {
                            Log.d(TAG,"Found FIFO characteristic!\n");
                            characteristicFifo = characteristic;
                            mBluetoothLeService.setCharacteristicNotification(characteristic, true);
                        } else if (uuid.equals(GattAttributes.UUID_CHARACTERISTIC_CREDITS)) {
                            Log.d(TAG,"Found Credits characteristic!\n");
                            mBluetoothLeService.setCharacteristicNotification(characteristic, false);
                            updateStatus();
                        }
                    }
                }
            });
        }

        @Override
        public void onGattDisconnected() {
            runOnUiThread(() -> {
                mConnectionState = ConnectionState.DISCONNECTED;
                invalidateOptionsMenu();
                updateStatus();
                rlProgress.setVisibility(View.GONE);
            });
        }

        @Override
        public void onGattConnected() {
            runOnUiThread(() -> {
                mConnectionState = ConnectionState.CONNECTED;
                invalidateOptionsMenu();
                updateStatus();
                rlProgress.setVisibility(View.GONE);
            });
        }
    }
}
