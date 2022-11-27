package reservation;

import java.io.*;
import java.util.Scanner;

public class doctorreservation // Child class(inherit from Mother class "Doctor") Inheritance
{
    private static boolean MainMenu = true;//abstraction
    private static boolean SubMenu = true;//abstraction

    public static void main(String[] args) throws IOException
    {
        Scanner input = new Scanner(System.in);
        doctor[] avdoc = new doctor[10];
        avdoc[0] = new doctor();
        avdoc[1] = new doctor();
        avdoc[2] = new doctor();
        avdoc[3] = new doctor();
        avdoc[4] = new doctor();
        avdoc[5] = new doctor();
        avdoc[6] = new doctor();
        avdoc[7] = new doctor();
        avdoc[8] = new doctor();
        avdoc[9] = new doctor();

        int roomNum = 0;
        initialise(avdoc);

        while (MainMenu)
        {
            while (SubMenu)
            {
                System.out.println("##########################################");
                System.out.println("   Welcome To Doctor Reservation System ");
                System.out.println("##########################################");
                System.out.println("A: Appoint a Doctor.");
                System.out.println("V: View all Doctors.");
                System.out.println("D: Delete patient.");
                System.out.println("F: Find Appointed doctor room from Patient name.");
                System.out.println("U: Update Patient name.");
                System.out.println("S: Store program data in to file.");
                System.out.println("L: Load program data from file.");
                System.out.println("##########################################");


                String Selection = input.next();
                Selection = Selection.toUpperCase();

                switch (Selection)
                {
                    case "A":
                        AppointDoc(avdoc, roomNum);
                        break;
                    case "V":
                        ViewAllAppoint(avdoc);
                        break;
                    case "D":
                        DeletePatient(avdoc, roomNum);
                        break;
                    case "F":
                        FindAppointedDoctorFromPatientName(avdoc);
                        break;
                    case "U":
                        UpdatePatientName(avdoc,roomNum);
                        break;
                    case "S":
                        StoreDataInToFile(avdoc);
                        break;
                    case "L":
                        LoadDataFromFile(avdoc);
                        break;
                    default:
                        System.out.println("Invalid Selection");
                        break;
                }
                System.out.println("Would you like to Select another Option\n1 ) Yes\n2 ) No");

                if (input.nextInt() == 1)
                {
                    SubMenu = true;
                }
                else
                {
                    SubMenu = false;
                }

            }
            SubMenu = true;
            System.out.println("Would You Like To Continue With The Program\n1 ) Yes\n2 ) No");

            if (input.nextInt() == 1)
            {
                MainMenu = true;
            }
            else
            {
                System.out.println("");
                System.exit(0);

            }
        }

    }
    private static void initialise(doctor[] avdoc)
    {
        for (int x = 0; x < avdoc.length; x++)
        {
            avdoc[x].setName("nobody");
        }
    }

    //Appoint a doctor
    private static void AppointDoc(doctor[] avdoc, int roomNum)
    {
        String roomName;
        int x;
        Scanner input = new Scanner(System.in);
        System.out.println("Enter Doctor id (1-10):");
        roomNum = input.nextInt() - 1;

        if (avdoc[roomNum].getName()!=("nobody") )//polymorphism getName
        {
            System.out.println("Sorry!This doctor already appointed.PLease enter the another Doctor id" );
            System.out.println("Enter doctor id (1-10):");
            roomNum = input.nextInt() - 1;
            System.out.println("Enter name for appointment " + (roomNum + 1) + " :");
            roomName = input.next();
            avdoc[roomNum].setName(roomName);//polymorphism setName
        }
        else
        {

            System.out.println("Enter name for appointment " + (roomNum + 1) + " :");
            roomName = input.next();
            avdoc[roomNum].setName(roomName);
        }



    }

    //ViewAllRooms
    private static void ViewAllAppoint(doctor[] avdoc)
    {
        for (int x = 0; x < avdoc.length; x++)
        {
            System.out.println("Doctor " + (x + 1) + " appointed by " + avdoc[x].getName());
        }
    }

    //delete customer
    private static void DeletePatient(doctor[] avdoc, int roomNum)
    {
        Scanner input = new Scanner(System.in);
        System.out.println("Enter doctor number to delete(1-10):");
        roomNum = input.nextInt() - 1;
        avdoc[roomNum].setName("nobody");
        System.out.println("Patient Entry Has Been Deleted...");
    }

    private static void FindAppointedDoctorFromPatientName(doctor[] avdoc)
    {
        Scanner input = new Scanner(System.in);
        String roomName;
        System.out.println("Enter name to Search for:");
        roomName = input.next();
        int x;
        boolean Checker = false;
        for (x = 0; x < avdoc.length; x++)
        {
            if (roomName.equals(avdoc[x].getName()))
            {
                System.out.println("Founded the Patient appointed with doctor id " + (x+1));
                Checker = true;
            }
        }
        if (Checker == false)
        {
            System.out.println("...There are no Doctor appointment with that name....");
        }
    }
    private static void UpdatePatientName(doctor[] avdoc, int roomNum)
    {
        Scanner input = new Scanner(System.in);
        String roomName;
        System.out.println("Enter Patient name for Update:");
        roomName = input.next();
        int x;
        boolean Checker = false;
        System.out.println("Enter new name for id " + (roomNum + 1) + " :");
        roomName = input.next();
        avdoc[roomNum].setName(roomName);
        System.out.println("Patient Name Has Been Updated Successfully");


    }


    private static void StoreDataInToFile(doctor[] avdoc) throws IOException
    {
        try (PrintWriter out = new PrintWriter(new FileWriter("C:\\Users\\DELL\\IdeaProjects\\Doctor_Reservation_System\\src\\output.txt")))
        {
            int x;
            for (x = 0; x < avdoc.length; x++)
            {
                out.println("Patient : " + avdoc[x].getName() + " get a appointed under doctor id : " + (x+1));
            }

        }
        System.out.println("All Patient Names have been Saved.");
    }

    private static void LoadDataFromFile(doctor[] avdoc) throws IOException
    {
        FileInputStream fs = new FileInputStream("C:\\Users\\DELL\\IdeaProjects\\Doctor_Reservation_System\\src\\input1.txt");
        BufferedReader br = new BufferedReader(new InputStreamReader(fs));
        for (int i = 0; i < avdoc.length; i++)
        {
            if(avdoc[i].getName()!=("nobody"))
            {
                continue;
            }
            else
            {
                avdoc[i].setName(br.readLine());
            }

        }
    }

    public static class doctor // Mother Class
    {

        private String mainName; //Encapsulation
        int guestsInRoom;

        public doctor()
        {
            mainName = "k";

        }

        public void setName(String aName)
        {

            mainName = aName;
        }

        public String getName()
        {
            return mainName;
        }
    }

}