package com.gwt.BLE.data;

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

    //private Bitmap photoPreview;

    static class Access {
        public boolean granted;
        public boolean oldGranted;
    }

    private HashMap<String, Access> accesses;

    public Visitor() { }

    public Visitor(String name, String description, byte[] descriptor) {
        this.name = name;
        this.description = description;
        this.descriptor = descriptor;
    }

    public Visitor(int id, String name, String description, byte[] descriptor) {
        this.id = id;
        this.name = name;
        this.description = description;
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

    public void setAccess(String address, boolean granted) {
        if (address == null) {
            Log.e (TAG, "Error: Unable to set access for null device");
            return;
        }

        if (this.accesses == null) {
            this.accesses = new HashMap<>();
        }
        Access a = this.accesses.get(address);
        if (a == null) {
            a = new Access();
            a.granted = granted;
            a.oldGranted = granted;
        }
        this.accesses.put(address, a);
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
