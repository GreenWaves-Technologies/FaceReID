package com.ublox.BLE.activities;

import android.annotation.TargetApi;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.os.Bundle;
import android.provider.Settings;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import com.gwt.BLE.activities.MainActivity;
import com.gwt.BLE.R;
import com.gwt.BLE.data.DataBaseHelper;
import com.gwt.BLE.data.Device;
import com.ublox.BLE.bluetooth.BluetoothCentral;
import com.ublox.BLE.bluetooth.BluetoothPeripheral;
import com.ublox.BLE.bluetooth.BluetoothScanner;
import com.ublox.BLE.interfaces.BluetoothDeviceRepresentation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static android.Manifest.permission.ACCESS_FINE_LOCATION;
import static android.bluetooth.BluetoothDevice.BOND_BONDED;
import static android.bluetooth.BluetoothDevice.BOND_BONDING;
import static android.bluetooth.BluetoothDevice.BOND_NONE;
import static android.content.pm.PackageManager.PERMISSION_GRANTED;

public class DevicesActivity extends Activity implements AdapterView.OnItemClickListener, BluetoothCentral.Delegate {

    private static final String TAG = DevicesActivity.class.getSimpleName();
    private static final int LOCATION_REQUEST = 255;
    private LeDeviceListAdapter mLeDeviceListAdapter;
    private Map<String, Device> reidDevices;
    private int favCnt;
    private BluetoothCentral scanner;
    private DataBaseHelper db;

    private static final int REQUEST_ENABLE_BT = 1;

    public static final String EXTRA_DEVICE = "device";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getActionBar().setTitle("");
        getActionBar().setLogo(R.drawable.logo);
        getActionBar().setDisplayUseLogoEnabled(true);

        setContentView(R.layout.activity_devices);

        reidDevices = new HashMap<>();

        // Should not be needed
        if (!getPackageManager().hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE)) {
            Toast.makeText(this, R.string.ble_not_supported, Toast.LENGTH_SHORT).show();
            finish();
        }

        scanner = null;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_devices, menu);
        if (scanner == null || scanner.getState() != BluetoothScanner.State.SCANNING) {
            menu.findItem(R.id.menu_stop).setVisible(false);
            menu.findItem(R.id.menu_scan).setVisible(true);
            menu.findItem(R.id.menu_progress).setActionView(null);
        } else {
            menu.findItem(R.id.menu_stop).setVisible(true);
            menu.findItem(R.id.menu_scan).setVisible(false);
            menu.findItem(R.id.menu_progress).setActionView(R.layout.actionbar_indeterminate_progress);
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.menu_scan:
                final BluetoothManager bluetoothManager = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
                BluetoothAdapter bluetoothAdapter = bluetoothManager.getAdapter();

                // Checks if Bluetooth is supported on the device.
                if (bluetoothAdapter == null) {
                    Toast.makeText(this, R.string.error_bluetooth_not_supported, Toast.LENGTH_SHORT).show();
                } else if (!bluetoothAdapter.isEnabled()) {
                    // If Bluetooth is not currently enabled, fire an intent to display a dialog
                    // asking the user to grant permission to enable it.
                    Intent enableBtIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
                    startActivityForResult(enableBtIntent, REQUEST_ENABLE_BT);
                } else {
                    if (scanner == null) {
                        scanner = new BluetoothScanner(bluetoothAdapter);
                        scanner.setDelegate(this);
                    }

                    mLeDeviceListAdapter.clear();
                    scanLeDevice(true);
                }
                break;
            case R.id.menu_stop:
                scanLeDevice(false);
                break;
            case R.id.menu_visitor_list:
                Intent intent = new Intent(this, MainActivity.class);
                startActivity(intent);
                break;
        }
        return true;
    }

    @Override
    protected void onResume() {
        super.onResume();

        // Initializes list view adapter.
        mLeDeviceListAdapter = new LeDeviceListAdapter();
        setListAdapter(mLeDeviceListAdapter);

        // Reopen DB
        db = new DataBaseHelper(getApplicationContext());

        // Reload known devices list
        ArrayList<Device> dbDevices = db.getAllDevices();
        for (int i = 0; i < dbDevices.size(); i++) {
            Device d = dbDevices.get(i);
            reidDevices.put(d.getAddress(), d);
        }
    }

    private void setListAdapter(BaseAdapter baseAdapter) {
        ListView lvDevices = findViewById(R.id.lvDevices);
        lvDevices.setAdapter(baseAdapter);
        lvDevices.setOnItemClickListener(this);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        // User chose not to enable Bluetooth.
        if (requestCode == REQUEST_ENABLE_BT && resultCode == Activity.RESULT_CANCELED) {
            finish();
            return;
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    @Override
    protected void onPause() {
        super.onPause();
        scanLeDevice(false);
        mLeDeviceListAdapter.clear();
        db.closeDB();
    }

    private void scanLeDevice(final boolean enable) {
        if (scanner == null) {
            return;
        }

        if (enable) {
            verifyPermissionAndScan();
        } else {
            scanner.stop();
        }
        invalidateOptionsMenu();
    }

    @TargetApi(23)
    private void verifyPermissionAndScan() {
        if (ContextCompat.checkSelfPermission(this, ACCESS_FINE_LOCATION) != PERMISSION_GRANTED) {
            requestPermissions(new String[]{ACCESS_FINE_LOCATION}, LOCATION_REQUEST);
            return;
        }

        if (Settings.Secure.getInt(getContentResolver(), Settings.Secure.LOCATION_MODE, Settings.Secure.LOCATION_MODE_OFF) == Settings.Secure.LOCATION_MODE_OFF) {
            Toast.makeText(this, R.string.location_permission_toast, Toast.LENGTH_LONG).show();
            Intent enableLocationIntent = new Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS);
            startActivity(enableLocationIntent);
            return;
        }

        scanner.scan(new ArrayList<>());
    }

    @Override
    public void onRequestPermissionsResult (int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode != LOCATION_REQUEST) return;

        if (grantResults.length > 0 && grantResults[0] == PERMISSION_GRANTED) {
            verifyPermissionAndScan();
        } else {
            Toast.makeText(this, R.string.location_permission_toast, Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
        if(mLeDeviceListAdapter.getCount() > position) {
            Log.d(TAG, "onListItemClick");
            BluetoothDeviceRepresentation device = mLeDeviceListAdapter.getDevice(position);

            scanner.stop();

            Intent intent = new Intent(this, MainActivity.class);
            Log.w(TAG, "Putting " + EXTRA_DEVICE + " " + (device == null));
            intent.putExtra(EXTRA_DEVICE, device);
            startActivity(intent);
        }
    }

    @Override
    public void centralChangedState(BluetoothCentral central) {
        invalidateOptionsMenu();
    }

    @Override
    public void centralFoundPeripheral(BluetoothCentral central, BluetoothPeripheral peripheral) {
        runOnUiThread(() -> mLeDeviceListAdapter.addDevice(((com.ublox.BLE.bluetooth.BluetoothDevice) peripheral).toUbloxDevice(), peripheral.rssi()));
    }

    // Adapter for holding devices found through scanning.
    private class LeDeviceListAdapter extends BaseAdapter {
        private ArrayList<BluetoothDeviceRepresentation> mLeDevices;
        private LayoutInflater mInflator;

        private HashMap<BluetoothDeviceRepresentation, Integer> mDevicesRssi = new HashMap<>();

        private LeDeviceListAdapter() {
            super();
            mLeDevices = new ArrayList<>();
            favCnt = 0;
            mInflator = DevicesActivity.this.getLayoutInflater();
        }

        private void addDevice(BluetoothDeviceRepresentation device, int rssi) {
            if (mDevicesRssi.containsKey(device)) {
                int oldRssi = mDevicesRssi.get(device);
                if (Math.abs(oldRssi - rssi) > 10) {
                    mDevicesRssi.put(device, rssi);
                    notifyDataSetChanged();
                }
            } else {
                mDevicesRssi.put(device, rssi);
                notifyDataSetChanged();
            }
            if (!mLeDevices.contains(device)) {
                Device d = reidDevices.get(device.getAddress());
                if (d != null && d.isFavourite()) {
                    mLeDevices.add(favCnt, device);
                    favCnt++;
                } else {
                    mLeDevices.add(device);
                }
                notifyDataSetChanged();
            }
        }

        private BluetoothDeviceRepresentation getDevice(int position) {
            return mLeDevices.get(position);
        }

        private void clear() {
            mLeDevices.clear();
            favCnt = 0;
        }

        @Override
        public int getCount() {
            return mLeDevices.size();
        }

        @Override
        public Object getItem(int i) {
            return mLeDevices.get(i);
        }

        @Override
        public long getItemId(int i) {
            return i;
        }

        @Override
        public View getView(final int i, View view, ViewGroup viewGroup) {
            BluetoothDeviceRepresentation device = mLeDevices.get(i);
            ViewHolder viewHolder;
            // General ListView optimization code.
            if (view == null) {
                view = mInflator.inflate(R.layout.listitem_device, null);
                viewHolder = new ViewHolder();
                viewHolder.deviceRssi = view.findViewById(R.id.device_rssi);
                viewHolder.deviceName = view.findViewById(R.id.device_name);
                viewHolder.deviceAddress = view.findViewById(R.id.device_address);
                viewHolder.deviceBonded = view.findViewById(R.id.device_bonded);
                viewHolder.imgRssi = view.findViewById(R.id.img_rssi);
                view.setTag(viewHolder);
            } else {
                viewHolder = (ViewHolder) view.getTag();
            }

            final View finalView = view;
            view.setOnClickListener(v -> onItemClick(null, finalView, i, i));

            final String deviceName = device.getName();
            final String deviceAddress = device.getAddress();
            if (deviceName != null && deviceName.length() > 0) {
                viewHolder.deviceName.setText(deviceName);
                viewHolder.deviceName.setVisibility(View.VISIBLE);
                viewHolder.deviceAddress.setTypeface(null, Typeface.NORMAL);
            }
            else {
                viewHolder.deviceName.setVisibility(View.INVISIBLE);
                //viewHolder.deviceAddress.setTypeface(null, Typeface.BOLD);
            }
            viewHolder.deviceAddress.setText(deviceAddress);
            updateBondedStateComponents(device, viewHolder);
            updateRssiComponents(device, viewHolder);

            return view;
        }

        private void updateBondedStateComponents(BluetoothDeviceRepresentation device, ViewHolder viewHolder) {
            switch(device.getBondState()) {
                case BOND_NONE:
                    viewHolder.deviceBonded.setVisibility(View.INVISIBLE);
                    break;
                case BOND_BONDING:
                    viewHolder.deviceBonded.setText(R.string.bonding_state);
                    viewHolder.deviceBonded.setVisibility(View.VISIBLE);
                    break;
                case BOND_BONDED:
                    viewHolder.deviceBonded.setText(R.string.bonded_state);
                    viewHolder.deviceBonded.setVisibility(View.VISIBLE);
                    break;
            }
        }

        private void updateRssiComponents(BluetoothDeviceRepresentation device, ViewHolder viewHolder) {
            final int rssi = mDevicesRssi.get(device);
            viewHolder.deviceRssi.setText(String.format("%s dBm", String.valueOf(rssi)));
            if(rssi <= -100) {
                viewHolder.imgRssi.setImageResource(R.drawable.signal_indicator_0);
            } else if (rssi < -85) {
                viewHolder.imgRssi.setImageResource(R.drawable.signal_indicator_1);
            } else if (rssi < -70) {
                viewHolder.imgRssi.setImageResource(R.drawable.signal_indicator_2);
            } else if (rssi < -55) {
                viewHolder.imgRssi.setImageResource(R.drawable.signal_indicator_3);
            } else {
                viewHolder.imgRssi.setImageResource(R.drawable.signal_indicator_4);
            }
        }

    }

    static class ViewHolder {
        ImageView imgRssi;
        TextView deviceRssi;
        TextView deviceName;
        TextView deviceAddress;
        TextView deviceBonded;
    }
}
