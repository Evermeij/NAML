/**
 * Created by bendesc on 21.aug.2017.
 */

//.eml part
import java.nio.file.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.io.*;
import javax.mail.*;
import javax.mail.internet.*;
import java.util.Date;

//.msg part
import java.util.List;
import com.auxilii.msgparser.*;
import com.auxilii.msgparser.Message;
import com.auxilii.msgparser.attachment.*;
import com.auxilii.msgparser.MsgParser;

public class ReadEmail {

    // EML sparsing code
    private String getTextFromMessage(MimeMessage message) throws MessagingException, IOException {
        String result = "";
        if (message.isMimeType("text/plain")) {
            result = message.getContent().toString();
        } else if (message.isMimeType("multipart/*")) {
            MimeMultipart mimeMultipart = (MimeMultipart) message.getContent();
            result = getTextFromMimeMultipart(mimeMultipart);
        }
        return result;
    }

    private String getTextFromMimeMultipart(
            MimeMultipart mimeMultipart)  throws MessagingException, IOException{
        String result = "";
        int count = mimeMultipart.getCount();
        for (int i = 0; i < count; i++) {
            BodyPart bodyPart = mimeMultipart.getBodyPart(i);
            if (bodyPart.isMimeType("text/plain")) {
                result = result + "\n" + bodyPart.getContent();
                break; // without break same text appears twice in my tests
            } else if (bodyPart.isMimeType("text/html")) {
             String html = (String) bodyPart.getContent();
              result = result + "\n" + org.jsoup.Jsoup.parse(html).text();
             }
            else if (bodyPart.getContent() instanceof MimeMultipart){
                result = result + getTextFromMimeMultipart((MimeMultipart)bodyPart.getContent());
            }
        }
        return result;
    }
    private String getTo(javax.mail.Message message) throws Exception{
        String to_addresses = "";
        Address[] recipients = message.getRecipients(javax.mail.Message.RecipientType.TO);
        for (Address address : recipients) {
            System.out.println("To : " +  address.toString() );
            to_addresses = to_addresses + " ," +address.toString();

        }
        return to_addresses;
    }

    public void sparseEml(String path,String emlFilename) throws Exception{

        File emlFile = new File(path+emlFilename);
        Properties props = System.getProperties();
        props.put("mail.host", "smtp.dummydomain.com");
        props.put("mail.transport.protocol", "smtp");

        Session mailSession = Session.getDefaultInstance(props, null);
        InputStream source = new FileInputStream(emlFile);
        MimeMessage message = new MimeMessage(mailSession, source);


        System.out.println("Subject : " + message.getSubject());
        System.out.println("From : " + message.getFrom()[0]);
        System.out.println("To : "+ getTo(message));
        System.out.println("Date: " + message.getSentDate() );

        System.out.println("Message Id : " +  message.getMessageID() );

        System.out.println("--------------");
        System.out.println("Body : " +  getTextFromMessage(message) );

        String toEmail = message.getMessageID();
        //String fromName = msg.getFromName();
        //String toName = msg.getToName();


        PrintWriter output_file = new PrintWriter("emails/processed/"+emlFilename.split("\\.", -1)[0]+"_eml_"+".txt");
        output_file.println("__Subject__ : " + message.getSubject());
        output_file.println("__From__ : " + message.getFrom()[0]);
        output_file.println("__To__ : " + getTo(message));
        output_file.println("__Date__ : " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(message.getSentDate() ) );
        output_file.println("__MessageId__ : " + message.getMessageID() );
        output_file.println("__Body__ : " +  getTextFromMessage(message) );
        output_file.close();
    }

    // MSG sparsing code
    public void sparseMsg(String path, String msgFilename) throws Exception{

        File msgFile = new File(path + msgFilename);

        MsgParser msgp = new MsgParser();

        Message msg = msgp.parseMsg(msgFile);

        String fromEmail = msg.getFromEmail();
        String toEmail = msg.getToEmail();
        String fromName = msg.getFromName();
        String toName = msg.getToName();
        String subject = msg.getSubject();
        String body = msg.getBodyText();

        //PrintWriter out = new PrintWriter("emails/processed/"+msgPath.split(".")[0]);

        System.out.println("From :" + fromName + " <" + fromEmail + ">");
        System.out.println("Subject :" + subject);
        System.out.println("To :" + toName + " <" + toEmail + ">");
        System.out.println("Date: " + msg.getDate());
        System.out.println(body);
        System.out.println("");

        PrintWriter output_file = new PrintWriter("emails/processed/" + msgFilename.split("\\.", -1)[0] + "_msg_" + ".txt");
        output_file.println("__Subject__ : " + subject);
        output_file.println("__From__ : " + fromEmail);
        output_file.println("__To__ : " + toEmail);
        output_file.println("__Date__ : " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(msg.getDate()));
        output_file.println("__MessageId__ : " + msg.getMessageId());
        output_file.println("__Body__ : " + body);
        output_file.close();


    }
    public static void main(String[] args) throws Exception{
        ReadEmail reademail = new ReadEmail();

    while(true) {
        //Sparse .eml mails
        File folder_eml = new File("emails/eml/");
        File[] listOfFiles_eml = folder_eml.listFiles();

        for (int i = 0; i < listOfFiles_eml.length; i++) {
            if (listOfFiles_eml[i].isFile()) {
                //System.out.println("File " + listOfFiles[i].getName());
                String textpath = "emails/eml/" + listOfFiles_eml[i].getName();
                System.out.println(textpath);
                try {
                    reademail.sparseEml("emails/eml/", listOfFiles_eml[i].getName());
                } catch (java.io.IOException e) {
                    System.out.println("Could not read file :" + listOfFiles_eml[i].getName());
                }
                Path path = Paths.get(textpath);
                try {

                    Files.delete(path);
                } catch (NoSuchFileException x) {
                    System.err.format("%s: no such" + " file or directory%n", path);
                } catch (DirectoryNotEmptyException x) {
                    System.err.format("%s not empty%n", path);
                } catch (IOException x) {
                    // File permission problems are caught here.
                    System.err.println(x);
                }
            } else if (listOfFiles_eml[i].isDirectory()) {
                System.out.println("Directory " + listOfFiles_eml[i].getName());
            }
        }
        //Sparse .msg mails
        File folder_msg = new File("emails/msg/");
        File[] listOfFiles_msg = folder_msg.listFiles();

        for (int i = 0; i < listOfFiles_msg.length; i++) {
            if (listOfFiles_msg[i].isFile()) {
                //System.out.println("File " + listOfFiles[i].getName());
                String textpath = "emails/msg/" + listOfFiles_msg[i].getName();
                System.out.println(textpath);
                try {
                    reademail.sparseMsg("emails/msg/", listOfFiles_msg[i].getName());
                } catch (java.io.IOException e) {
                    System.out.println("Could not read file :" + listOfFiles_msg[i].getName());
                }
                Path path = Paths.get(textpath);
                try {

                    Files.delete(path);
                } catch (NoSuchFileException x) {
                    System.err.format("%s: no such" + " file or directory%n", path);
                } catch (DirectoryNotEmptyException x) {
                    System.err.format("%s not empty%n", path);
                } catch (IOException x) {
                    // File permission problems are caught here.
                    System.err.println(x);
                }
            } else if (listOfFiles_msg[i].isDirectory()) {
                System.out.println("Directory " + listOfFiles_msg[i].getName());
            }
        }
    }
    }



}