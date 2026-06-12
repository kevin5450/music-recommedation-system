package com.example.demo.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "firstfivesongs")
public class FirstFiveSongs {
    @Id
    private String id;
    private String title;
    private String artist;

    public FirstFiveSongs() {}

    public FirstFiveSongs(String title, String artist) {
        this.title = title;
        this.artist = artist;
    }

    // Getter & Setter
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getArtist() { return artist; }
    public void setArtist(String artist) { this.artist = artist; }
}
