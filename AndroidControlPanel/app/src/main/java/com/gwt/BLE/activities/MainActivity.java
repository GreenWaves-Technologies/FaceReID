package com.gwt.BLE.activities;

import android.app.ActionBar;
import android.app.Activity;
import android.app.AlertDialog;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattService;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.text.InputType;
import android.text.TextUtils;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Adapter;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.gwt.BLE.R;
import com.ublox.BLE.interfaces.BluetoothDeviceRepresentation;
import com.ublox.BLE.services.BluetoothLeService;
import com.ublox.BLE.services.BluetoothLeServiceReceiver;
import com.ublox.BLE.utils.ConnectionState;
import com.ublox.BLE.utils.GattAttributes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.UUID;

import static com.ublox.BLE.services.BluetoothLeService.ITEM_TYPE_NOTIFICATION;
import static com.ublox.BLE.services.BluetoothLeService.ITEM_TYPE_READ;
import static com.ublox.BLE.services.BluetoothLeService.ITEM_TYPE_WRITE;
import static java.lang.Math.min;

public class MainActivity extends Activity {

    private static final String TAG = "MyBleActivity";

    public static final String EXTRA_DEVICE = "device";
    public static final String EXTRA_REMOTE = "remote";

    private static final byte BLE_ACK = (byte)0x33;

    private static final byte BLE_READ = (byte)0x10;
    private static final byte BLE_GET_NAME = (byte)0x11;
    private static final byte BLE_GET_PHOTO = (byte)0x12;
    private static final byte BLE_GET_DESCRIPTOR = (byte)0x13;
    private static final byte BLE_REMOVE = (byte)0x14;

    private static final byte BLE_WRITE = (byte)0x20;
    private static final byte BLE_SET_NAME = (byte)0x21;
    private static final byte BLE_SET_DESCRIPTOR = (byte)0x22;

    private static final byte BLE_EXIT = (byte)0x55;

    private byte previousBleRequest;
    private byte[] currentUserPhotoToRead = new byte[128*128];
    private int currentUserPhotoRead = 0;
    private byte[] currentUserDescriptorToRead = new byte[512*2];
    private int currentUserDescriptorRead = 0;

    private boolean isRemoteMode;
    private TextView tvStatus;
    private RelativeLayout rlProgress;
    private ListView androidListView;

    private BluetoothDeviceRepresentation mDevice;

    private BluetoothLeService mBluetoothLeService;
    private static ConnectionState mConnectionState = ConnectionState.DISCONNECTED;

    private BluetoothGattCharacteristic characteristicFifo;

    public void onServiceConnected() {
        if (!mBluetoothLeService.initialize(this)) {
            finish();
        }
    }

    public final MyBroadcastReceiver mGattUpdateReceiver = new MyBroadcastReceiver();

    class PersonProfile {
      public String name;
      public byte[] photoData;
      public byte[] descriptor;
      public Bitmap photoPreview;
    };

    ArrayList<PersonProfile> strangers;
    private PersonProfile currentUserToRead = new PersonProfile();
    private PersonProfile currentUserToWrite;
    private int currentUserToWriteIdx = -1;

    class PeopleListAdapter extends BaseAdapter {
        Context context;
        LayoutInflater inflater;
        ArrayList<PersonProfile> people;

        public PeopleListAdapter(Context context, ArrayList<PersonProfile> people)
        {
            this.context = context;
            this.people = people;
            inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        }

        public ArrayList<PersonProfile> getPeople() {
            return people;
        }

        @Override
        public int getCount() {
            return people.size();
        }

        @Override
        public Object getItem(int position) {
            return people.get(position);
        }

        @Override
        public long getItemId(int position) {
            return 0;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            View view = convertView;
            if(view == null){
                view = inflater.inflate(R.layout.listitem_person, parent, false);
            }

            PersonProfile person = people.get(position);

            TextView personTextView = (TextView) view.findViewById(R.id.person_name);
            personTextView.setText(person.name);

            ImageView personPreview = (ImageView) view.findViewById(R.id.person_photo);

            final int pixCount = 128 * 128;
            int[] intGreyBuffer = new int[pixCount];
            for(int i=0; i < pixCount; i++)
            {
                int greyValue = (int)person.photoData[i] & 0xff;
                intGreyBuffer[i] = 0xff000000 | (greyValue << 16) | (greyValue << 8) | greyValue;
            }
            person.photoPreview = Bitmap.createBitmap(intGreyBuffer, 128, 128, Bitmap.Config.ARGB_8888);

            personPreview.setImageBitmap(person.photoPreview);

            return view;
        }
    };

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

        tvStatus = findViewById(R.id.tvStatus);
        rlProgress = findViewById(R.id.rlProgress);

        final Intent intent = getIntent();
        isRemoteMode = intent.hasExtra(EXTRA_REMOTE);
        updateStatus();

        strangers = new ArrayList<>();
        PeopleListAdapter adapter = new PeopleListAdapter(this, strangers);
        androidListView = (ListView) findViewById(R.id.person_list);
        androidListView.setAdapter(adapter);

        androidListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                currentUserToWriteIdx = position;
                currentUserToWrite = strangers.get(position);
                AlertDialog.Builder builder = new AlertDialog.Builder(view.getContext());
                builder.setTitle("Person name");

                final EditText input = new EditText(view.getContext());
                // Specify the type of input expected; this, for example, sets the input as a password, and will mask the text
                input.setInputType(InputType.TYPE_CLASS_TEXT);
                builder.setView(input);

                builder.setPositiveButton("Remember", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        String newName = input.getText().toString();
                        if(!newName.isEmpty()) {
                            currentUserToWrite.name = newName;
                            final Adapter a = androidListView.getAdapter();
                            if (a instanceof BaseAdapter) {
                                ((BaseAdapter) a).notifyDataSetChanged();
                            }

                            Log.d(TAG, "Sending request to add new person");
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_WRITE});
                            rlProgress.setVisibility(View.VISIBLE);
                            previousBleRequest = BLE_WRITE;
                        }
                    }
                });

                builder.setNegativeButton("Drop", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.cancel();
                        strangers.remove(currentUserToWriteIdx);
                        final Adapter a = androidListView.getAdapter();
                        if (a instanceof BaseAdapter) {
                            ((BaseAdapter) a).notifyDataSetChanged();
                        }
                        currentUserToWriteIdx = -1;
                        currentUserToWrite = null;
                    }
                });

                builder.show();
            }
        });

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
                    menu.findItem(R.id.menu_refresh_people).setVisible(true);
                    break;
                case CONNECTING:
                    Log.d(TAG, "Create menu in Connecting mode");
                    menu.findItem(R.id.menu_connect).setVisible(false);
                    menu.findItem(R.id.menu_disconnect).setVisible(false);
                    menu.findItem(R.id.menu_refresh_people).setVisible(false);
                    break;
                case DISCONNECTED:
                    Log.d(TAG, "Create menu in Disconnected mode");
                    menu.findItem(R.id.menu_connect).setVisible(true);
                    menu.findItem(R.id.menu_disconnect).setVisible(false);
                    menu.findItem(R.id.menu_refresh_people).setVisible(false);
                    break;
                case BLE_EXCHANGE:
                    Log.d(TAG, "Create menu in Exchange mode");
                    menu.findItem(R.id.menu_connect).setVisible(false);
                    menu.findItem(R.id.menu_disconnect).setVisible(false);
                    menu.findItem(R.id.menu_refresh_people).setVisible(false);
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
                if((mConnectionState == ConnectionState.CONNECTED) || (mConnectionState==ConnectionState.BLE_EXCHANGE)) {
                    mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_EXIT});
                }
                mBluetoothLeService.disconnect();
                updateStatus();
                rlProgress.setVisibility(View.VISIBLE);
                return true;
            case R.id.menu_refresh_people:
                if(mBluetoothLeService != null) {
                    strangers.clear();
                    final Adapter a = androidListView.getAdapter();
                    if (a instanceof BaseAdapter) {
                        ((BaseAdapter)a).notifyDataSetChanged();
                    }
                    Log.d(TAG, "Starting people enumeration on device");
                    mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_READ});
                    previousBleRequest = BLE_READ;
                    mConnectionState = ConnectionState.BLE_EXCHANGE;
                    invalidateOptionsMenu();
                    updateStatus();
                    rlProgress.setVisibility(View.VISIBLE);
                }
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

                if(type == ITEM_TYPE_NOTIFICATION) {
                    Log.d(TAG, "previousBleRequest was " + previousBleRequest);

                    switch (previousBleRequest) {
                        case BLE_READ:
                            Log.d(TAG, "previousBleRequest == BLE_READ");
                            Log.d(TAG, "Response code: " + data[0]);
                            if (data[0] == BLE_ACK) {
                                Log.d(TAG, "Sending name request");
                                mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_GET_NAME});
                                previousBleRequest = BLE_GET_NAME;
                            }
                            else if (data[0] == 0)
                            {
                                mConnectionState = ConnectionState.CONNECTED;
                                runOnUiThread(() -> {
                                    invalidateOptionsMenu();
                                    updateStatus();
                                    rlProgress.setVisibility(View.GONE);
                                });
                            }
                            break;
                        case BLE_GET_NAME:
                            Log.d(TAG, "previousBleRequest == BLE_GET_NAME");
                            Log.d(TAG, "Name " + new String(data) + " got, sending BLE_GET_PHOTO request");
                            currentUserToRead.name = new String(data);
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_GET_PHOTO});
                            previousBleRequest = BLE_GET_PHOTO;
                            currentUserPhotoRead = 0;
                            break;
                        case BLE_GET_PHOTO:
                            Log.d(TAG, "previousBleRequest == BLE_GET_PHOTO");
                            for (int i = 0; i < data.length; i++) {
                                currentUserPhotoToRead[currentUserPhotoRead + i] = data[i];
                            }
                            currentUserPhotoRead += data.length;
                            Log.d(TAG, "Received " + currentUserPhotoRead + " bytes from " + currentUserPhotoToRead.length);
                            if (currentUserPhotoRead >= currentUserPhotoToRead.length) {
                                currentUserToRead.photoData = currentUserPhotoToRead.clone();
                                currentUserPhotoRead = 0;
                                Log.d(TAG, "Photo got, sending BLE_GET_DESCRIPTOR request");
                                mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_GET_DESCRIPTOR});
                                previousBleRequest = BLE_GET_DESCRIPTOR;
                            }
                            else
                            {
                                // data is sent in chunks by 1024 bytes. New request is needed to get the next portion
                                if(currentUserPhotoRead % 1024 == 0) {
                                    Log.d(TAG, "Requesting new chunk of data");
                                    mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_GET_PHOTO});
                                }
                            }
                            break;
                        case BLE_GET_DESCRIPTOR:
                            Log.d(TAG, "previousBleRequest == BLE_GET_DESCRIPTOR");
                            for (int i = 0; i < data.length; i++) {
                                currentUserDescriptorToRead[currentUserDescriptorRead + i] = data[i];
                            }
                            currentUserDescriptorRead += data.length;
                            Log.d(TAG, "Received " + currentUserDescriptorRead + " bytes from " + currentUserDescriptorToRead.length);
                            if (currentUserDescriptorRead >= currentUserDescriptorToRead.length) {
                                currentUserToRead.descriptor = currentUserDescriptorToRead.clone();
                                currentUserDescriptorRead = 0;
                                Log.d(TAG, "Descriptor got, sending BLE_REMOVE request");
                                mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_REMOVE});
                                previousBleRequest = BLE_REMOVE;
                            }
                            break;
                        case BLE_REMOVE:
                            strangers.add(currentUserToRead);
                            currentUserToRead = new PersonProfile();
                            currentUserPhotoToRead = new byte[128*128];
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_READ});
                            previousBleRequest = BLE_READ;
                            runOnUiThread(() -> {
                                final Adapter a = androidListView.getAdapter();
                                if (a instanceof BaseAdapter) {
                                    ((BaseAdapter)a).notifyDataSetChanged();
                                }
                            });
                            break;
                        case BLE_WRITE:
                            Log.d(TAG, "previousBleRequest == BLE_WRITE");
                            Log.d(TAG, "Response code: " + data[0]);
                            if (data[0] == BLE_ACK)
                            {
                                Log.d(TAG, "Sending request to set name");
                                byte[] name;
                                if(currentUserToWrite.name.length() >= 16)
                                {
                                    Log.d(TAG, "Name is longer than 16 symbols, getting 16 first letters");
                                    name = currentUserToWrite.name.substring(0,16).getBytes();
                                }
                                else
                                {
                                    Log.d(TAG, "Name is shorter than 16 letters, adding zeros.");
                                    name =  new byte[16];
                                    System.arraycopy(currentUserToWrite.name.getBytes(),0, name, 0, currentUserToWrite.name.length());
                                    for(int i = currentUserToWrite.name.length(); i < 16; i++)
                                    {
                                        name[i] = 0;
                                    }
                                }
                                byte[] request = new byte[17];
                                request[0] = BLE_SET_NAME;
                                System.arraycopy(name, 0, request, 1, 16);
                                {
                                    Log.d(TAG, "Sending request bytes");
                                    mBluetoothLeService.writeCharacteristic(characteristicFifo, request);
                                    Log.d(TAG, "Name bytes are sent");
                                }
                                previousBleRequest = BLE_SET_NAME;
                            }
                            else
                            {
                                Log.d(TAG, "Device responded with non BLE_ACK code: " + data[0]);
                            }
                            break;
                        case BLE_SET_NAME:
                            Log.d(TAG, "previousBleRequest == SET_NAME");
                            Log.d(TAG, "Response code: " + data[0]);
                            if (data[0] == BLE_ACK) {
                                mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_SET_DESCRIPTOR});
                                int chunksize = 20;
                                int packetsToSend = (currentUserToWrite.descriptor.length + 19) / chunksize;
                                for(int i = 0; i < packetsToSend; i++) {
                                    byte[] tmp = Arrays.copyOfRange(currentUserToWrite.descriptor, i*chunksize, min(i*chunksize + chunksize, 1024));
                                    mBluetoothLeService.writeCharacteristic(characteristicFifo, tmp);
                                }
                                previousBleRequest = BLE_SET_DESCRIPTOR;
                            }
                            break;
                        case BLE_SET_DESCRIPTOR:
                            Log.d(TAG, "previousBleRequest == SET_NAME");
                            Log.d(TAG, "Response code: " + data[0]);
                            if (data[0] == BLE_ACK) {
                                // exclude from list
                                strangers.remove(currentUserToWriteIdx);
                                currentUserToWriteIdx = -1;
                                currentUserToWrite = null;

                                runOnUiThread(() -> {
                                    rlProgress.setVisibility(View.GONE);
                                    final Adapter a = androidListView.getAdapter();
                                    if (a instanceof BaseAdapter) {
                                        ((BaseAdapter)a).notifyDataSetChanged();
                                    }
                                });

                                if(strangers.size() == 0)
                                {
                                    mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_EXIT});
                                }
                            }
                            break;
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
