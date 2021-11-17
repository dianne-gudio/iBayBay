package com.example.ibaybayversion1;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class OutputActivity extends AppCompatActivity {
    private Button doneBtn;
    private TextView outputResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_output);

        //Assign variables
        doneBtn = (Button) findViewById(R.id.doneBtn);
        outputResult = (TextView) findViewById(R.id.outputResult);

        //"Done" Button
        doneBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                backToHome();
            }
        });

        //Python API: This will start Python.
        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        /*
        Create Python instance.
         */
        Python py = Python.getInstance();

        /*
        Then, create Python object. -----------TO BE EDITED, THIS IS JUST A SAMPLE----------
         */
        PyObject pyobj = py.getModule("script"); //Opens the Python file named "script.py" inside "python" folder
        PyObject obj = pyobj.callAttr("main"); //Calls the function named "main" inside "script.py"

        outputResult.setText(obj.toString()); //Displays the text from Python "main" function to the app textview
    }

    public void backToHome(){
        Intent i = new Intent(this, MainActivity.class); //Opens Main Menu screen
        startActivity(i);
    }
}