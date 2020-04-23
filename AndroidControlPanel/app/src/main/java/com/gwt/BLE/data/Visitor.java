package com.gwt.BLE.data;

import android.graphics.Bitmap;
import android.util.Log;

import java.util.HashMap;

public class Visitor {

    private static final String TAG = "Visitor";

    private int id;

    private String name;
    private String oldName;

    private String description;

    private byte[] photoData;
    private byte[] descriptor;

    private HashMap<String, Access> accesses;

    public static class Access {
        public boolean granted;
        public boolean oldGranted;
    }

    public Visitor() { }

    public Visitor(int id, String name, String oldName, String description, byte[] photoData, byte[] descriptor) {
        this.id = id;
        this.name = name;
        this.oldName = oldName;
        this.description = description;
        this.photoData = photoData;
        this.descriptor = descriptor;
    }

    // Setters
    public void setId(int id) {
        this.id = id;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public void setOldName(String oldName) {
        this.oldName = oldName;
    }

    public void setDescriptor(byte[] descriptor) {
        this.descriptor = descriptor;
    }

    public void setPhotoData(byte[] photoData) {
        this.photoData = photoData;
    }

    public void setAccess(String address, Access access) {
        if (address == null) {
            Log.e (TAG, "Error: Unable to set access for null device");
            return;
        }

        if (this.accesses == null) {
            this.accesses = new HashMap<>();
        }

        this.accesses.put(address, access);
    }

    public void setAccesses(HashMap<String, Access> accesses) {
        this.accesses = accesses;
    }

    // Getters
    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String getOldName() {
        return oldName;
    }

    public String getDescription() {
        return description;
    }

    public byte[] getDescriptor() {
        return descriptor;
    }

    public byte[] getPhotoData() {
        return photoData;
    }

    public Bitmap getPhoto() {
        if (photoData == null) {
            return null;
        }

        final int pixCount = 128 * 128;
        int[] intGreyBuffer = new int[pixCount];
        for (int i = 0; i < pixCount; i++) {
            int greyValue = (int) photoData[i] & 0xff;
            intGreyBuffer[i] = 0xff000000 | (greyValue << 16) | (greyValue << 8) | greyValue;
        }
        return Bitmap.createBitmap(intGreyBuffer, 128, 128, Bitmap.Config.ARGB_8888);
    }

    public Access getAccess(String address) {
        if (address == null) {
            Log.e (TAG, "Error: Unable to get access for null device");
            return null;
        }

        if (this.accesses == null) {
            this.accesses = new HashMap<>();
        }
        return this.accesses.get(address);
    }

    public HashMap<String, Access> getAccesses() {
        return accesses;
    }
}
