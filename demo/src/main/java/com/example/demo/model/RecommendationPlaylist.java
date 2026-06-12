package com.example.demo.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "title_recommendation")
public class RecommendationPlaylist {
    @Id
    private String id;
    private String title;
    private String artist;
    private String album;

    public RecommendationPlaylist() {}

    public RecommendationPlaylist(String title, String artist, String album) {
        this.title = title;
        this.artist = artist;
        this.album = album;
    }

    public RecommendationPlaylist(String title, String artist) {
        this(title, artist, null);
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getArtist() { return artist; }
    public void setArtist(String artist) { this.artist = artist; }

    public String getAlbum() { return album; }
    public void setAlbum(String album) { this.album = album; }
}
