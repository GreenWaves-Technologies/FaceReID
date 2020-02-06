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
import com.gwt.BLE.data.DataBaseHelper;
import com.gwt.BLE.data.Device;
import com.gwt.BLE.data.Visitor;
import com.ublox.BLE.interfaces.BluetoothDeviceRepresentation;
import com.ublox.BLE.services.BluetoothLeService;
import com.ublox.BLE.utils.ConnectionState;
import com.ublox.BLE.utils.GattAttributes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Timer;
import java.util.TimerTask;
import java.util.UUID;

import static com.ublox.BLE.services.BluetoothLeService.ITEM_TYPE_NOTIFICATION;
import static com.ublox.BLE.services.BluetoothLeService.ITEM_TYPE_READ;
import static com.ublox.BLE.services.BluetoothLeService.ITEM_TYPE_WRITE;
import static java.lang.Math.min;

public class MainActivity extends Activity {

    private static final String TAG = "MyBleActivity";

    public static final String EXTRA_DEVICE = "device";

    private static final byte BLE_ACK = 0x33;

    private static final byte BLE_READ_STRANGER = 0x10;
    private static final byte BLE_GET_STRANGER_NAME = 0x11;
    private static final byte BLE_GET_STRANGER_PHOTO = 0x12;
    private static final byte BLE_GET_STRANGER_DESCRIPTOR = 0x13;
    private static final byte BLE_DROP_STRANGER = 0x14;

    private static final byte BLE_READ_VISITOR = 0x15;
    private static final byte BLE_GET_VISITOR_NAME = 0x16;
    private static final byte BLE_GET_VISITOR_DESCRIPTOR = 0x17;
    private static final byte BLE_DROP_VISITOR = 0x18;

    private static final byte BLE_WRITE = 0x20;
    private static final byte BLE_SET_NAME = 0x21;
    private static final byte BLE_SET_DESCRIPTOR = 0x22;

    private static final byte BLE_EXIT = 0x55;

    private static final byte BLE_HEART_BEAT = 0x56;

    private static final int hbInterval = 10000; // 10s
    private Timer hbTimer;

    private byte currentBleRequest;
    private byte[] currentUserPhotoToRead = new byte[128*128];
    private int currentUserPhotoRead = 0;
    private byte[] currentUserDescriptorToRead = new byte[512*2];
    private int currentUserDescriptorRead = 0;

    private TextView tvStatus;
    private RelativeLayout rlProgress;
    private ListView androidListView;

    private BluetoothDeviceRepresentation mDevice;

    private BluetoothLeService mBluetoothLeService;
    private static ConnectionState mConnectionState = ConnectionState.DISCONNECTED;

    private BluetoothGattCharacteristic characteristicFifo;

    private DataBaseHelper db;
    private Device dbDevice;

    public void onServiceConnected() {
        if (!mBluetoothLeService.initialize(this)) {
            finish();
        }
    }

    public final MyBroadcastReceiver mGattUpdateReceiver = new MyBroadcastReceiver();

    ArrayList<Visitor> visitors;
    ArrayList<Integer> visitorsPermitted;
    private Visitor currentUserToRead = new Visitor();
    private Visitor currentUserToWrite;
    private int currentUserToWriteIdx = -1;

    class PeopleListAdapter extends BaseAdapter {
        Context context;
        LayoutInflater inflater;
        ArrayList<Visitor> people;
        ArrayList<Integer> peoplePermitted; // Indicates that person has access to currently connected device

        private PeopleListAdapter(Context context, ArrayList<Visitor> people, ArrayList<Integer> peoplePermitted)
        {
            this.context = context;
            this.people = people;
            this.peoplePermitted = peoplePermitted;
            inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        }

        @Override
        public int getCount() {
            if (people == null) {
                return 0;
            } else {
                return people.size();
            }
        }

        @Override
        public Object getItem(int position) {
            if (people == null) {
                return  null;
            } else {
                return people.get(position);
            }
        }

        @Override
        public long getItemId(int position) {
            return 0;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            View view = convertView;
            if (view == null) {
                view = inflater.inflate(R.layout.listitem_person, parent, false);
            }

            Visitor person = people.get(position);

            TextView nameTextView = view.findViewById(R.id.person_name);
            nameTextView.setText(person.getName());

            TextView descriptionTextView = view.findViewById(R.id.person_description);
            descriptionTextView.setText(person.getDescription());

            ImageView personPreview = view.findViewById(R.id.person_photo);
            Bitmap photoPreview = person.getPhoto();
            if (photoPreview != null) {
                personPreview.setImageBitmap(photoPreview);
            } else {
                personPreview.setImageResource(R.drawable.ic_unknown_person);
            }

            ImageView accessIndicator = view.findViewById(R.id.person_indicator);
            if (peoplePermitted.contains(person.getId())) {
                accessIndicator.setVisibility(View.VISIBLE);
            } else {
                accessIndicator.setVisibility(View.INVISIBLE);
            }

            return view;
        }
    }

    private void updateStatus() {
        switch (mConnectionState) {
            case CONNECTING:
                tvStatus.setText(R.string.status_connecting);
                break;
            case CONNECTED:
                tvStatus.setText(R.string.status_connected);
                break;
            case DISCONNECTING:
                tvStatus.setText(R.string.status_disconnecting);
                break;
            case DISCONNECTED:
                tvStatus.setText(R.string.status_disconnected);
                break;
            case BLE_EXCHANGE:
                tvStatus.setText(R.string.status_loading);
                break;
        }
    }

    private void startHBTimer() {
        hbTimer = new Timer();
        TimerTask hbTask = new TimerTask() {
            public void run() {
                mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_HEART_BEAT});
            }
        };
        hbTimer.schedule(hbTask, hbInterval / 2, hbInterval);
    }

    private void stopHBTimer() {
        if (hbTimer != null) {
            hbTimer.cancel();
        }
    }

    private String CString2String(byte[] data) {
        int i;
        for (i = 0; i < data.length; i++) {
            if (data[i] == 0x00) {
                break;
            }
        }
        return new String(data, 0, i);
    }

    @Override
    protected void onResume() {
        super.onResume();

        db = new DataBaseHelper(getApplicationContext());
        if (mDevice != null) {
            dbDevice = db.getDevice(mDevice.getAddress());
            if (dbDevice != null) {
                dbDevice.setName(mDevice.getName());
            } else {
                dbDevice = new Device(mDevice.getName(), mDevice.getAddress());
                db.createDevice(dbDevice);
            }
        }

        ArrayList<Visitor> dbVisitors = db.getAllVisitors(/*mDevice.getAddress()*/);
        if (visitors.isEmpty()) {
            visitors.addAll(dbVisitors);
        } else { // merge two lists
            int listSize = visitors.size();
            for (Visitor v : dbVisitors) {
                int i;
                for (i = 0; i < listSize; i++) {
                    if (visitors.get(i).getId() == v.getId()) {
                        break;
                    }
                }
                if (i == listSize) {
                    visitors.add(v);
                }
            }
        }

        if (mDevice != null && mBluetoothLeService != null) {
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
        if (mDevice != null) {
            try {
                stopHBTimer();
                mBluetoothLeService.disconnect();
                mBluetoothLeService.close();
                mConnectionState = ConnectionState.DISCONNECTED;
                mBluetoothLeService.unregister();
            } catch (Exception ignore) {
            }

            db.updateDevice(dbDevice);
        }

        db.closeDB();

        invalidateOptionsMenu();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        final ActionBar actionBar = getActionBar();
        actionBar.setTitle("");
        actionBar.setLogo(R.drawable.logo);
        actionBar.setDisplayUseLogoEnabled(true);

        setContentView(R.layout.activity_main);

        tvStatus = findViewById(R.id.tvStatus);
        rlProgress = findViewById(R.id.rlProgress);

        updateStatus();

        visitors = new ArrayList<>();
        visitorsPermitted = new ArrayList<>();

        PeopleListAdapter adapter = new PeopleListAdapter(this, visitors, visitorsPermitted);
        androidListView = findViewById(R.id.person_list);
        androidListView.setAdapter(adapter);

        androidListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                currentUserToWriteIdx = position;
                currentUserToWrite = visitors.get(position);
                AlertDialog.Builder builder = new AlertDialog.Builder(view.getContext());
                builder.setTitle("Person name");

                final EditText input = new EditText(view.getContext());
                // Specify the type of input expected; this, for example, sets the input as a password, and will mask the text
                input.setInputType(InputType.TYPE_CLASS_TEXT);
                input.setText(currentUserToWrite.getName());
                builder.setView(input);

                builder.setPositiveButton("Remember", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        String newName = input.getText().toString();
                        if (!newName.isEmpty()) {
                            currentUserToWrite.setName(newName);
                            final Adapter a = androidListView.getAdapter();
                            if (a instanceof BaseAdapter) {
                                ((BaseAdapter) a).notifyDataSetChanged();
                            }

                            db.updateVisitor(currentUserToWrite);

                            if (mDevice != null) {
                                currentUserToWrite.setAccess(mDevice.getAddress(), true);
                                db.updateAccess(currentUserToWrite.getId(), mDevice.getAddress(), currentUserToWrite.getAccess(mDevice.getAddress()));

                                Log.d(TAG, "Sending request to add new person");
                                mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_WRITE});
                                rlProgress.setVisibility(View.VISIBLE);
                                currentBleRequest = BLE_WRITE;
                            }
                        }
                    }
                });

                builder.setNegativeButton("Drop", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.cancel();

                        visitorsPermitted.remove(Integer.valueOf(currentUserToWrite.getId()));
                        visitors.remove(currentUserToWriteIdx);
                        db.deleteVisitor(currentUserToWrite);

                        if (mDevice != null) {
                            Log.d(TAG, "Sending request to drop a person");
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_DROP_VISITOR});
                            int chunkSize = 20;
                            int packetsToSend = (currentUserToWrite.getDescriptor().length + chunkSize - 1) / chunkSize;
                            for (int i = 0; i < packetsToSend; i++) {
                                byte[] tmp = Arrays.copyOfRange(currentUserToWrite.getDescriptor(), i * chunkSize, min(i * chunkSize + chunkSize, 1024));
                                mBluetoothLeService.writeCharacteristic(characteristicFifo, tmp);
                            }
                            currentBleRequest = BLE_DROP_VISITOR;
                        }

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

        final Intent intent = getIntent();
        mDevice = intent.getParcelableExtra(EXTRA_DEVICE);
        if (mDevice != null) {
            connectToDevice();

            final String name = mDevice.getName();
            if (!TextUtils.isEmpty(name)) {
                actionBar.setTitle(name);
            } else {
                actionBar.setTitle(mDevice.getAddress());
            }
        }

        actionBar.setDisplayShowCustomEnabled(true);
    }

    private void connectToDevice() {
        // get the information from the device scan
        mBluetoothLeService = new BluetoothLeService();
        onServiceConnected();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_connected, menu);
        if (mDevice != null) {
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
                case DISCONNECTING:
                    Log.d(TAG, "Create menu in Disconnecting mode");
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

            if (dbDevice.isFavourite()) {
                menu.findItem(R.id.menu_favourite).setIcon(R.drawable.ic_star_black_24dp);
            } else {
                menu.findItem(R.id.menu_favourite).setIcon(R.drawable.ic_star_border_black_24dp);
            }
            menu.findItem(R.id.menu_favourite).setVisible(true);
        } else {
            Log.d(TAG, "Create menu in Detached mode");
            menu.findItem(R.id.menu_connect).setVisible(false);
            menu.findItem(R.id.menu_disconnect).setVisible(false);
            menu.findItem(R.id.menu_refresh_people).setVisible(false);
            menu.findItem(R.id.menu_favourite).setVisible(false);
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
                stopHBTimer();
                if ((mConnectionState == ConnectionState.CONNECTED) || (mConnectionState == ConnectionState.BLE_EXCHANGE)) {
                    mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_EXIT});
                    currentBleRequest = BLE_EXIT;
                    mConnectionState = ConnectionState.DISCONNECTING;
                } else {
                    mBluetoothLeService.disconnect();
                }

                invalidateOptionsMenu();
                updateStatus();
                rlProgress.setVisibility(View.VISIBLE);
                return true;
            case R.id.menu_refresh_people:
                if(mBluetoothLeService != null) {
                    visitorsPermitted.clear();
                    final Adapter a = androidListView.getAdapter();
                    if (a instanceof BaseAdapter) {
                        ((BaseAdapter)a).notifyDataSetChanged();
                    }
                    Log.d(TAG, "Starting people enumeration on device");
                    mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_READ_VISITOR});
                    currentBleRequest = BLE_READ_VISITOR;
                    mConnectionState = ConnectionState.BLE_EXCHANGE;
                    invalidateOptionsMenu();
                    updateStatus();
                    //rlProgress.setVisibility(View.VISIBLE);
                }
                return true;
            case R.id.menu_favourite:
                if (dbDevice.isFavourite()) {
                    dbDevice.setFavourite(false);
                    item.setIcon(R.drawable.ic_star_border_black_24dp);
                } else {
                    dbDevice.setFavourite(true);
                    item.setIcon(R.drawable.ic_star_black_24dp);
                }
                return true;
            case R.id.menu_devices:
            case android.R.id.home:
                onBackPressed();
                return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private class MyBroadcastReceiver implements BluetoothLeService.Receiver {
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
            String typeString;
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
                Log.d(TAG, "BLE request was " + currentBleRequest);

                switch (currentBleRequest) {
                    case BLE_READ_STRANGER:
                        Log.d(TAG, "currentBleRequest == BLE_READ_STRANGER");
                        Log.d(TAG, "Response code: " + data[0]);
                        if (data[0] == BLE_ACK) {
                            Log.d(TAG, "Sending stranger name request");
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_GET_STRANGER_NAME});
                            currentBleRequest = BLE_GET_STRANGER_NAME;
                        } else if (data[0] == 0) {
                            mConnectionState = ConnectionState.CONNECTED;
                            runOnUiThread(() -> {
                                invalidateOptionsMenu();
                                updateStatus();
                                rlProgress.setVisibility(View.GONE);
                            });
                        }
                        break;
                    case BLE_GET_STRANGER_NAME: {
                        Log.d(TAG, "currentBleRequest == BLE_GET_STRANGER_NAME");
                        String name = CString2String(data);
                        Log.d(TAG, "Name " + name + " got, sending BLE_GET_STRANGER_PHOTO request");
                        currentUserToRead.setName(name);
                        mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_GET_STRANGER_PHOTO});
                        currentBleRequest = BLE_GET_STRANGER_PHOTO;
                        currentUserPhotoRead = 0;
                        break;
                    }
                    case BLE_GET_STRANGER_PHOTO:
                        Log.d(TAG, "currentBleRequest == BLE_GET_STRANGER_PHOTO");
                        System.arraycopy(data, 0, currentUserPhotoToRead, currentUserPhotoRead, data.length);
                        currentUserPhotoRead += data.length;
                        Log.d(TAG, "Received " + currentUserPhotoRead + " bytes from " + currentUserPhotoToRead.length);
                        if (currentUserPhotoRead >= currentUserPhotoToRead.length) {
                            currentUserToRead.setPhotoData(currentUserPhotoToRead.clone());
                            currentUserPhotoRead = 0;
                            Log.d(TAG, "Photo got, sending BLE_GET_STRANGER_DESCRIPTOR request");
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_GET_STRANGER_DESCRIPTOR});
                            currentBleRequest = BLE_GET_STRANGER_DESCRIPTOR;
                        } else {
                            // data is sent in chunks by 1024 bytes. New request is needed to get the next portion
                            if(currentUserPhotoRead % 1024 == 0) {
                                Log.d(TAG, "Requesting new chunk of data");
                                mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_GET_STRANGER_PHOTO});
                            }
                        }
                        break;
                    case BLE_GET_STRANGER_DESCRIPTOR:
                        Log.d(TAG, "currentBleRequest == BLE_GET_STRANGER_DESCRIPTOR");
                        System.arraycopy(data, 0, currentUserDescriptorToRead, currentUserDescriptorRead, data.length);
                        currentUserDescriptorRead += data.length;
                        Log.d(TAG, "Received " + currentUserDescriptorRead + " bytes from " + currentUserDescriptorToRead.length);
                        if (currentUserDescriptorRead >= currentUserDescriptorToRead.length) {
                            currentUserToRead.setDescriptor(currentUserDescriptorToRead.clone());
                            currentUserDescriptorRead = 0;
                            Log.d(TAG, "Descriptor got, sending BLE_DROP_STRANGER request");
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_DROP_STRANGER});
                            currentBleRequest = BLE_DROP_STRANGER;
                        }
                        break;
                    case BLE_DROP_STRANGER:
                        Log.d(TAG, "currentBleRequest == BLE_DROP_STRANGER");
                        if (db.createVisitor(currentUserToRead) >= 0) {
                            visitors.add(currentUserToRead);
                        }

                        currentUserToRead = new Visitor();
                        currentUserPhotoToRead = new byte[128*128];
                        mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_READ_STRANGER});
                        currentBleRequest = BLE_READ_STRANGER;
                        runOnUiThread(() -> {
                            final Adapter a = androidListView.getAdapter();
                            if (a instanceof BaseAdapter) {
                                ((BaseAdapter)a).notifyDataSetChanged();
                            }
                        });
                        break;
                    case BLE_READ_VISITOR:
                        Log.d(TAG, "currentBleRequest == BLE_READ_VISITOR");
                        Log.d(TAG, "Response code: " + data[0]);
                        if (data[0] == BLE_ACK) {
                            Log.d(TAG, "Sending visitor name request");
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_GET_VISITOR_NAME});
                            currentBleRequest = BLE_GET_VISITOR_NAME;
                        } else if (data[0] == 0) {
                            // All visitors are loaded, continue with strangers
                            Log.d(TAG, "Sending read stranger request");
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_READ_STRANGER});
                            currentBleRequest = BLE_READ_STRANGER;
                        }
                        break;
                    case BLE_GET_VISITOR_NAME: {
                        Log.d(TAG, "currentBleRequest == BLE_GET_VISITOR_NAME");
                        String name = CString2String(data);
                        Log.d(TAG, "Name " + name + " got, sending BLE_GET_VISITOR_DESCRIPTOR request");
                        currentUserToRead.setName(name);
                        mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_GET_VISITOR_DESCRIPTOR});
                        currentBleRequest = BLE_GET_VISITOR_DESCRIPTOR;
                        break;
                    }
                    case BLE_GET_VISITOR_DESCRIPTOR:
                        Log.d(TAG, "currentBleRequest == BLE_GET_VISITOR_DESCRIPTOR");
                        System.arraycopy(data, 0, currentUserDescriptorToRead, currentUserDescriptorRead, data.length);
                        currentUserDescriptorRead += data.length;
                        Log.d(TAG, "Received " + currentUserDescriptorRead + " bytes from " + currentUserDescriptorToRead.length);
                        if (currentUserDescriptorRead >= currentUserDescriptorToRead.length) {
                            currentUserToRead.setDescriptor(currentUserDescriptorToRead.clone());
                            currentUserDescriptorRead = 0;
                            Log.d(TAG, "Descriptor got, adding a visitor");

                            // Try to find visitor in DB visitors
                            int i;
                            for (i = 0; i < visitors.size(); i++) {
                                Visitor v = visitors.get(i);
                                if (Arrays.equals(currentUserToRead.getDescriptor(), v.getDescriptor())) {
                                    visitorsPermitted.add(v.getId());
                                    break;
                                }
                            }

                            if (i >= visitors.size()) { // visitor is not in DB
                                db.createVisitor(currentUserToRead);
                                visitors.add(currentUserToRead);
                                visitorsPermitted.add(currentUserToRead.getId());
                            }

                            currentUserToRead = new Visitor();
                            currentUserPhotoToRead = new byte[128*128];
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_READ_VISITOR});
                            currentBleRequest = BLE_READ_VISITOR;
                            runOnUiThread(() -> {
                                final Adapter a = androidListView.getAdapter();
                                if (a instanceof BaseAdapter) {
                                    ((BaseAdapter)a).notifyDataSetChanged();
                                }
                            });
                        }
                        break;
                    case BLE_DROP_VISITOR:
                        Log.d(TAG, "currentBleRequest == BLE_DROP_VISITOR");
                        Log.d(TAG, "Response code: " + data[0]);
                        break;
                    case BLE_WRITE:
                        Log.d(TAG, "currentBleRequest == BLE_WRITE");
                        Log.d(TAG, "Response code: " + data[0]);
                        if (data[0] == BLE_ACK)
                        {
                            Log.d(TAG, "Sending request to set name");
                            byte[] name;
                            if (currentUserToWrite.getName().length() >= 16)
                            {
                                Log.d(TAG, "Name is longer than 16 symbols, getting 16 first letters");
                                name = currentUserToWrite.getName().substring(0, 16).getBytes();
                            } else {
                                Log.d(TAG, "Name is shorter than 16 letters, adding zeros.");
                                name = new byte[16];
                                System.arraycopy(currentUserToWrite.getName().getBytes(), 0, name, 0, currentUserToWrite.getName().length());
                                for (int i = currentUserToWrite.getName().length(); i < 16; i++)
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
                            currentBleRequest = BLE_SET_NAME;
                        } else {
                            Log.d(TAG, "Device responded with non BLE_ACK code: " + data[0]);
                        }
                        break;
                    case BLE_SET_NAME:
                        Log.d(TAG, "currentBleRequest == SET_NAME");
                        Log.d(TAG, "Response code: " + data[0]);
                        if (data[0] == BLE_ACK) {
                            mBluetoothLeService.writeCharacteristic(characteristicFifo, new byte[]{BLE_SET_DESCRIPTOR});
                            int chunkSize = 20;
                            int packetsToSend = (currentUserToWrite.getDescriptor().length + chunkSize - 1) / chunkSize;
                            for(int i = 0; i < packetsToSend; i++) {
                                byte[] tmp = Arrays.copyOfRange(currentUserToWrite.getDescriptor(), i*chunkSize, min(i*chunkSize + chunkSize, 1024));
                                mBluetoothLeService.writeCharacteristic(characteristicFifo, tmp);
                            }
                            currentBleRequest = BLE_SET_DESCRIPTOR;
                        }
                        break;
                    case BLE_SET_DESCRIPTOR:
                        Log.d(TAG, "currentBleRequest == SET_DESCRIPTOR");
                        Log.d(TAG, "Response code: " + data[0]);
                        if (data[0] == BLE_ACK) {
                            visitorsPermitted.add(currentUserToWrite.getId());

                            currentUserToWriteIdx = -1;
                            currentUserToWrite = null;

                            runOnUiThread(() -> {
                                rlProgress.setVisibility(View.GONE);
                                final Adapter a = androidListView.getAdapter();
                                if (a instanceof BaseAdapter) {
                                    ((BaseAdapter)a).notifyDataSetChanged();
                                }
                            });
                        }
                        break;
                    case BLE_EXIT:
                        Log.d(TAG, "currentBleRequest == EXIT");
                        Log.d(TAG, "Response code: " + data[0]);
                        if (data[0] == BLE_ACK) {
                            mBluetoothLeService.disconnect();
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
            stopHBTimer();
        }

        @Override
        public void onGattConnected() {
            runOnUiThread(() -> {
                mConnectionState = ConnectionState.CONNECTED;
                visitorsPermitted.clear();
                invalidateOptionsMenu();
                updateStatus();
                rlProgress.setVisibility(View.GONE);
            });
            startHBTimer();
        }
    }
}
