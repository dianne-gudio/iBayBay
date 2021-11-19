package com.example.ibaybayversion1;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class OpenChartActivity extends AppCompatActivity {
    private Button backBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_open_chart);

        //Assign variables
        backBtn = (Button) findViewById(R.id.backBtn);

        //"Back To Menu" button
        backBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                backToHome();
            }
        });
    }

    public void backToHome(){
        Intent i = new Intent(this, MainActivity.class); //Opens Main Menu screen
        startActivity(i);
    }
}