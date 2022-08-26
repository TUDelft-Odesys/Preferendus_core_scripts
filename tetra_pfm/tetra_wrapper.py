"""
Code to send to and receive data from teh Tetra API endpoint. Code is an python adaptation of the Matlab code written
by Dmitry Zhilyaev, TU Delft (D.Zhilyaev@tudelft.nl).

Credentials are needed for the API. The credentials are not free to share (at the moment at least).

Copyright (c) Harold van Heukelum, 2021
"""
import pathlib
import xml.etree.ElementTree as Et

import urllib3
from numpy import array, ndarray
from requests import post
from requests.exceptions import HTTPError

HERE = pathlib.Path(__file__).parent


class TetraSolver:
    """
    Class to interact with the web version of Tetra. Builds the XML used as input for Tetra and returns the output of
    Tetra as an array with floats.
    """

    def __init__(self):
        # the certificate of the server has expired, hence we need to exclude a warning
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.xml_file = f'{HERE}/tetra_in.xml'  # path to XML that is sent to Tetra server

        self.headers = {'Content-Type': 'text/xml; charset=utf-8'}  # headers for post requests
        self.url = 'https://choicerobot.com:7997/SMXWebServices/Solve.php'  # URL of Tetra Server

        self.user = 'tudelft'
        with open(f'{HERE}/credentials.txt', 'r') as f:
            self.password = f.read()

    def _indent(self, elem, level=0):
        """
        Credits to Erick M. Sprengel for this method's code. Source: https://bit.ly/3IpXZHP

        Method sets correct indent for XML file, so that it is better readable

        :param elem: root of your XML file
        :param level: indent level
        :return: None
        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
        return

    def _make_xml(self, w , p):
        """
        Method to build the XML file that is sent to the Tetra server. The basis of the document is the template
        document template1.xml

        It returns the XML tree directly. This means, the XML-file does not have to be saved and then reloaded to use it

        :param p: Population
        :param w: Weights
        :return: XML ElementTree
        """

        criteria_count = len(p)  # count the number of criteria C
        try:
            pop_size = len(p[0])
        except TypeError:  # if only one alternative is added
            pop_size = 1

        start_alternatives = 3  # skip RefA1 and RefA2
        end_alternatives = start_alternatives + pop_size

        tree = Et.parse(f'{HERE}/xml-templates/template1.xml')  # load template
        root = tree.getroot()
        criteria_element = root.find('Criterion')
        comment = Et.Element('Comment')  # since comment is empty everywhere, we can create it once

        for i in range(start_alternatives, end_alternatives):  # add alternatives A to XML tree root
            addition = Et.Element('Alternative')
            addition.append(comment)
            addition.attrib = {'Name': f'A{i - 2}', 'Type': '0'}
            root.insert(i, addition)

        addition = Et.Element('Criterion')  # add criterion start again, since its place is overwritten
        addition.append(comment)
        addition.attrib = {'Name': 'Criteria', 'IsPreferenceIncreasing': '1'}

        for j in range(1, criteria_count + 1):  # add scores of alternatives per criterion C
            criterion = Et.Element('Criterion')
            criterion.attrib = {'Name': f'C{j}', 'IsPreferenceIncreasing': '1'}  # add criterion
            criterion.append(comment)
            measurement = Et.Element('Measurement')  # add measurement element. Unused, but required
            measurement.attrib = {'Name': 'Ratings', 'Type': '0', 'DecisionMaker': 'Default User', 'SubType': '0'}
            measurement.append(comment)
            for k in range(start_alternatives, end_alternatives):  # add scores per alternatives to element C
                ruler_value = Et.Element('RulerValue')
                try:
                    ruler_value.attrib = {'Name': f'A{k - 2}', 'Value': f'{p[j - 1][k - 3]}'}
                except IndexError:  # if p is no n-by-m array
                    ruler_value.attrib = {'Name': f'A{k - 2}', 'Value': f'{p[k - 3]}'}
                measurement.append(ruler_value)
            criterion.append(measurement)
            criteria_element.append(criterion)

        weights = Et.Element('Measurement')  # add weights to XML as measurement element
        weights.attrib = {'Name': 'Criteria Weights', 'Type': '1', 'DecisionMaker': 'Default User', 'SubType': '0'}
        weights.append(comment)
        for n in range(1, criteria_count + 1):
            ruler_value = Et.Element('RulerValue')
            ruler_value.attrib = {'Name': f'C{n}', 'Value': f'{w[n - 1]}'}  # specify weight w per criterion C
            weights.append(ruler_value)
        criteria_element.append(weights)

        self._indent(root)  # set correct indent for file
        tree.write(self.xml_file, encoding='utf-8', xml_declaration=True)  # save XML tree as XML-file
        return tree  # return XML tree

    def request(self, w, p):
        """
        Method that handles the communication with the Tetra server. Gathers xml ElementTree from method make_xml and
        returns an array with the values that are returned from Tetra.

        :param w: weight per criterion
        :param p: array that contains the individual scores per criterion per member of the population
        :return: list with aggregated scores for all alternatives
        """
        self.assertion_tests(w, p)  # assert w & p have no foreseen mistakes
        if not p:
            return []  # if p is empty, return an empty list too

        # create XML and store it as a string to variable xml_tree
        xml_tree = Et.tostring(self._make_xml(w=w, p=p).getroot(), encoding='utf8', method='xml')

        try:  # try reaching the Tetra server and post the XML-file
            r = post(self.url, data=xml_tree, auth=(self.user, self.password), headers=self.headers, verify=False,
                     timeout=10)
            r.raise_for_status()
        except HTTPError as err:  # if an error is raised by the server or request service, raise an HTTPError
            raise HTTPError(f'HTTP Error occurred: {err}')

        values = list()
        tree = Et.fromstring(r.content)  # make XML tree from returned content from the server
        self._indent(tree)  # set correct indents
        Et.ElementTree(tree).write(f'{HERE}/tetra_out.xml', encoding='utf-8', xml_declaration=True)  # save XML-file
        for items in tree.findall('Alternative'):  # find all Alternative-elements in the XMl-tree
            values.append(-float(items.attrib.get('Value')))  # extract values from XML and save it to the list 'values'

        assert len(values) == len(p[0])  # assert no data elements are lost in the process
        return values

    @staticmethod
    def assertion_tests(w, p):
        """Assertion tests to check integrity of the data input"""
        assert len(w) == len(array(p)), 'Lists of weights and functions should be equal'
        assert round(sum(w), 4) == 1, f'Sum of weights should be equal to 1. They are now sum({w})={sum(w)}.'
        return
