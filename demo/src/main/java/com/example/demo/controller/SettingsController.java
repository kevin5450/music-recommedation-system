package com.example.demo.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class SettingsController {

    @GetMapping("/settings")
    public String settingsPage() {
        return "settings";
    }

    @GetMapping("/settings-account")
    public String settingsAccount(Model model) {
        return "settings-account";
    }

    @GetMapping("/settings-etc")
    public String settingsEtc() {
        return "settings-etc";
    }
}