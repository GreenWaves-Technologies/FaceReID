package com.gwt.BLE.data;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class DataBaseHelper extends SQLiteOpenHelper {

    private static final String TAG = "DBHelper";

    private static final String DATABASE_NAME = "reid.db";
    private static final int DATABASE_VERSION = 1;

    private static final String TABLE_DEVICES = "devices";
    private static final String TABLE_VISITORS = "visitors";
    private static final String TABLE_ACCESS = "access";

    // TABLE_DEVICES
    private static final String KEY_DEVICE_NAME = "name";
    private static final String KEY_DEVICE_MACADDR = "mac";
    private static final String KEY_DEVICE_FAVOURITE = "favourite";
    // TABLE_VISITORS
    private static final String KEY_VISITOR_ID = "id";
    private static final String KEY_VISITOR_PHOTO = "photo";
    private static final String KEY_VISITOR_NAME = "name";
    private static final String KEY_VISITOR_OLD_NAME = "name_dev";
    private static final String KEY_VISITOR_DESCRIPTION = "description";
    private static final String KEY_VISITOR_DESCRIPTOR = "descriptor";
    // TABLE_ACCESS
    private static final String KEY_ACCESS_MACADDR = "mac";
    private static final String KEY_ACCESS_VISITOR_ID = "pid";
    private static final String KEY_ACCESS_GRANTED = "granted";
    private static final String KEY_ACCESS_OLD_GRANTED = "granted_dev";

    public DataBaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);

        Log.i(TAG, "Open DB: " + DATABASE_NAME);
        SQLiteDatabase db = this.getWritableDatabase();
        db.setForeignKeyConstraintsEnabled(true);
    }


    @Override
    public void onCreate(SQLiteDatabase db) {
        Log.i(TAG, "Create DB: " + DATABASE_NAME);

        db.execSQL(
            "CREATE TABLE " + TABLE_DEVICES + " (" +
                KEY_DEVICE_NAME + " text, " +
                KEY_DEVICE_MACADDR + "  NOT NULL PRIMARY KEY CHECK (length(" + KEY_DEVICE_MACADDR + ") >= 12), " +
                KEY_DEVICE_FAVOURITE + " boolean" +
            ")"
        );

        db.execSQL(
            "CREATE TABLE " + TABLE_VISITORS + " (" +
                KEY_VISITOR_ID + " integer PRIMARY KEY," +
                KEY_VISITOR_PHOTO + " blob UNIQUE," +
                KEY_VISITOR_NAME + " text CHECK (" + KEY_VISITOR_NAME + " <> \"\")," +
                KEY_VISITOR_OLD_NAME + " text," +
                KEY_VISITOR_DESCRIPTION + " text," +
                KEY_VISITOR_DESCRIPTOR + " blob NOT NULL UNIQUE CHECK (length(" + KEY_VISITOR_DESCRIPTOR + ") = 1024)" +
            ")"
        );

        db.execSQL(
            "CREATE TABLE " + TABLE_ACCESS + " (" +
                KEY_ACCESS_MACADDR + " text NOT NULL REFERENCES " + TABLE_DEVICES + " (" + KEY_DEVICE_MACADDR + "), " +
                KEY_ACCESS_VISITOR_ID + " integer NOT NULL REFERENCES " + TABLE_VISITORS + " (" + KEY_VISITOR_ID + "), " +
                KEY_ACCESS_GRANTED + " boolean, " +
                KEY_ACCESS_OLD_GRANTED + " boolean, " +
                "PRIMARY KEY( " + KEY_ACCESS_MACADDR + ", " + KEY_ACCESS_VISITOR_ID + " )" +
            ")"
        );
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        Log.i(TAG, "Upgrade DB " + DATABASE_NAME + " from v" + oldVersion + " to v" + newVersion);

        db.execSQL("DROP TABLE IF EXISTS " + TABLE_ACCESS);
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_VISITORS);
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_DEVICES);

        onCreate(db);
    }

    public void closeDB() {
        Log.i(TAG, "Close DB: " + DATABASE_NAME);
        SQLiteDatabase db = this.getReadableDatabase();
        if (db != null && db.isOpen()) {
            db.close();
        }
    }

    public long createDevice(Device device) {
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        values.put(KEY_DEVICE_NAME, device.getName());
        values.put(KEY_DEVICE_MACADDR, device.getAddress());
        values.put(KEY_DEVICE_FAVOURITE, device.isFavourite());

        return db.insert(TABLE_DEVICES, null, values);
    }

    public Device getDevice(String address) {
        SQLiteDatabase db = this.getReadableDatabase();

        Cursor c = db.query(TABLE_DEVICES, null, KEY_DEVICE_MACADDR + " = ?", new String[] { address }, null, null, null);
        if (!c.moveToFirst()) {
            return null;
        }

        Device d = new Device();
        d.setAddress(c.getString(c.getColumnIndex(KEY_DEVICE_MACADDR)));
        d.setName(c.getString(c.getColumnIndex(KEY_DEVICE_NAME)));
        d.setFavourite(c.getInt(c.getColumnIndex(KEY_DEVICE_FAVOURITE)) > 0);
        c.close();

        return d;
    }

    public Map<String, Device> getAllDevices() {
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor c = db.query(TABLE_DEVICES, null, null, null, null, null, null);

        Map<String, Device> devices = new HashMap<>();
        if (!c.moveToFirst()) {
            return devices;
        }

        do {
            Device d = new Device();
            d.setAddress(c.getString(c.getColumnIndex(KEY_DEVICE_MACADDR)));
            d.setName(c.getString(c.getColumnIndex(KEY_DEVICE_NAME)));
            d.setFavourite(c.getInt(c.getColumnIndex(KEY_DEVICE_FAVOURITE)) > 0);

            devices.put(d.getAddress(), d);
        } while(c.moveToNext());

        c.close();

        return devices;
    }

    public int updateDevice(Device device) {
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        values.put(KEY_DEVICE_NAME, device.getName());
        values.put(KEY_DEVICE_FAVOURITE, device.isFavourite());

        return db.update(TABLE_DEVICES, values, KEY_DEVICE_MACADDR + " = ?", new String[] { device.getAddress() });
    }

    public void deleteDevice(Device device) {
        SQLiteDatabase db = this.getWritableDatabase();

        db.delete(TABLE_ACCESS, KEY_ACCESS_MACADDR + " = ?", new String[] { device.getAddress() });
        db.delete(TABLE_DEVICES, KEY_DEVICE_MACADDR + " = ?", new String[] { device.getAddress() });
    }

    public long createVisitor(Visitor visitor) {
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        // values.put(KEY_VISITOR_ID, visitor.getId()); // autogenerated
        values.put(KEY_VISITOR_NAME, visitor.getName());
        values.put(KEY_VISITOR_OLD_NAME, visitor.getOldName());
        values.put(KEY_VISITOR_DESCRIPTION, visitor.getDescription());
        values.put(KEY_VISITOR_PHOTO, visitor.getPhotoData());
        values.put(KEY_VISITOR_DESCRIPTOR, visitor.getDescriptor());

        long rowid = db.insert(TABLE_VISITORS, null, values); // rowid == visitor id
        // db.insertWithOnConflict(TABLE_VISITORS, null, values, CONFLICT_REPLACE);
        if (rowid < 0) {
            Log.e(TAG, "Error: unable to create visitor " + visitor.getName());
            return rowid;
        }

        visitor.setId((int)rowid);
        HashMap<String, Visitor.Access> accesses = visitor.getAccesses();
        if (accesses != null) {
            for (String addr : accesses.keySet()) {
                createAccess(visitor.getId(), addr, accesses.get(addr));
            }
        }

        return rowid;
    }

    public Visitor getVisitor(long id) {
        SQLiteDatabase db = this.getReadableDatabase();

        Cursor c = db.query(TABLE_VISITORS, null, KEY_VISITOR_ID + " = ?", new String[] { String.valueOf(id) }, null, null, null);
        if (!c.moveToFirst()) {
            return null;
        }

        Visitor v = new Visitor();
        v.setId(c.getInt(c.getColumnIndex(KEY_VISITOR_ID)));
        v.setName(c.getString(c.getColumnIndex(KEY_VISITOR_NAME)));
        v.setOldName(c.getString(c.getColumnIndex(KEY_VISITOR_OLD_NAME)));
        v.setDescription(c.getString(c.getColumnIndex(KEY_VISITOR_DESCRIPTION)));
        v.setPhotoData(c.getBlob(c.getColumnIndex(KEY_VISITOR_PHOTO)));
        v.setDescriptor(c.getBlob(c.getColumnIndex(KEY_VISITOR_DESCRIPTOR)));
        c.close();

        v.setAccesses(getAccess(v.getId()));

        return v;
    }

    public ArrayList<Visitor> getAllVisitors(/* TODO: currently connected device */) {
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor c = db.query(TABLE_VISITORS, null, null, null, null, null, null /* TODO: display visistors known on device first ?*/);

        ArrayList<Visitor> visitors = new ArrayList<>();
        if (!c.moveToFirst()) {
            return visitors;
        }

        do {
            Visitor v = new Visitor();
            v.setId(c.getInt(c.getColumnIndex(KEY_VISITOR_ID)));
            v.setName(c.getString(c.getColumnIndex(KEY_VISITOR_NAME)));
            v.setOldName(c.getString(c.getColumnIndex(KEY_VISITOR_OLD_NAME)));
            v.setDescription(c.getString(c.getColumnIndex(KEY_VISITOR_DESCRIPTION)));
            v.setPhotoData(c.getBlob(c.getColumnIndex(KEY_VISITOR_PHOTO)));
            v.setDescriptor(c.getBlob(c.getColumnIndex(KEY_VISITOR_DESCRIPTOR)));
            v.setAccesses(getAccess(v.getId()));

            visitors.add(v);
        } while(c.moveToNext());

        c.close();

        return visitors;
    }

    public int updateVisitor(Visitor visitor) {
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        values.put(KEY_VISITOR_NAME, visitor.getName());
        values.put(KEY_VISITOR_OLD_NAME, visitor.getOldName());
        values.put(KEY_VISITOR_DESCRIPTION, visitor.getDescription());
        values.put(KEY_VISITOR_PHOTO, visitor.getPhotoData());
        values.put(KEY_VISITOR_DESCRIPTOR, visitor.getDescriptor());

        // updateAccess() should be called separately

        return db.update(TABLE_VISITORS, values, KEY_VISITOR_ID + " = ?", new String[] { String.valueOf(visitor.getId()) });
    }

    public void deleteVisitor(Visitor visitor) {
        SQLiteDatabase db = this.getWritableDatabase();

        db.delete(TABLE_ACCESS, KEY_ACCESS_VISITOR_ID + " = ?", new String[] { String.valueOf(visitor.getId()) });
        db.delete(TABLE_VISITORS, KEY_VISITOR_ID + " = ?", new String[] { String.valueOf(visitor.getId()) });
    }

    public long createAccess(int visitorId, String address, Visitor.Access access) {
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        values.put(KEY_ACCESS_VISITOR_ID, visitorId);
        values.put(KEY_ACCESS_MACADDR, address);
        values.put(KEY_ACCESS_GRANTED, access.granted);
        values.put(KEY_ACCESS_OLD_GRANTED, access.oldGranted);

        return db.insert(TABLE_ACCESS, null, values);
    }

    public HashMap<String, Visitor.Access> getAccess(int visitorId) {
        SQLiteDatabase db = this.getReadableDatabase();

        Cursor c = db.query(TABLE_ACCESS, null, KEY_ACCESS_VISITOR_ID + " = ?", new String[] { String.valueOf(visitorId)}, null, null, null);

        HashMap<String, Visitor.Access> accesses = new HashMap<>();
        if (!c.moveToFirst()) {
            return accesses;
        }

        do {
            String address = c.getString(c.getColumnIndex(KEY_ACCESS_MACADDR));
            Visitor.Access a = new Visitor.Access();
            a.granted = c.getInt(c.getColumnIndex(KEY_ACCESS_GRANTED)) > 0;
            a.oldGranted = c.getInt(c.getColumnIndex(KEY_ACCESS_OLD_GRANTED)) > 0;

            accesses.put(address, a);
        } while(c.moveToNext());
        c.close();

        return accesses;
    }

    public int updateAccess(int visitorId, String address, Visitor.Access access) {
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        values.put(KEY_ACCESS_GRANTED, access.granted);
        values.put(KEY_ACCESS_OLD_GRANTED, access.oldGranted);

        return db.update(TABLE_ACCESS, values, KEY_ACCESS_VISITOR_ID + " = ? AND " + KEY_ACCESS_MACADDR + " = ?", new String[] { String.valueOf(visitorId), address });
    }

    public void deleteAccess(int visitorId, String address) {
        SQLiteDatabase db = this.getWritableDatabase();

        db.delete(TABLE_ACCESS, KEY_ACCESS_VISITOR_ID + " = ? AND " + KEY_ACCESS_MACADDR + " = ?", new String[] { String.valueOf(visitorId), address });
    }
}
