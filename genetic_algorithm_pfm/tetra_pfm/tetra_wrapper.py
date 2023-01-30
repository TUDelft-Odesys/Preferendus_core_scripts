"""
Code to send to and receive data from teh Tetra API endpoint. Code is an python adaptation of the Matlab code written
by Dmitry Zhilyaev, TU Delft (D.Zhilyaev@tudelft.nl).

Credentials are needed for the API. The credentials are not free to share (at the moment at least).

Copyright (c) Harold van Heukelum, 2021
"""
import pathlib
import xml.etree.ElementTree as Et

import urllib3
from numpy import array
from requests import post
from requests.exceptions import HTTPError

HERE = pathlib.Path(__file__).parent


class TetraSolver:
    """
    Class to interact with the web version of Tetra. Builds the XML used as input for Tetra and returns the output of
    Tetra as an array with floats.
    """

    def __init__(self):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.xml_file = f'{HERE}/tetra_in.xml'

        self.headers = {'Content-Type': 'text/xml; charset=utf-8'}
        self.url = 'https://choicerobot.com:7997/SMXWebServices/Solve.php'

        self.user = 'tudelft'
        with open(f'{HERE}/credentials.txt', 'r') as f:
            self.password = f.read()

    def _indent(self, elem, level=0):
        """
        Credits to Erick M. Sprengel for this method's code. Source: https://bit.ly/3IpXZHP

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

    def _make_xml(self, w, p):
        """
        Method to build the XML file that is send to the Tetra server. The basis of the document is the template
        document.

        :param p: Population
        :param w: Weights
        :return: XML ElementTree
        """

        criteria_count = len(p)
        try:
            pop_size = len(p[0])
        except TypeError:
            pop_size = 1

        start_alternatives = 3
        end_alternatives = start_alternatives + pop_size

        tree = Et.parse(f'{HERE}/xml-templates/template1.xml')
        root = tree.getroot()
        criteria_element = root.find('Criterion')
        comment = Et.Element('Comment')

        for i in range(start_alternatives, end_alternatives):
            addition = Et.Element('Alternative')
            addition.append(comment)
            addition.attrib = {'Name': f'A{i - 2}', 'Type': '0'}
            root.insert(i, addition)

        addition = Et.Element('Criterion')
        addition.append(comment)
        addition.attrib = {'Name': 'Criteria', 'IsPreferenceIncreasing': '1'}

        for j in range(1, criteria_count + 1):
            criterion = Et.Element('Criterion')
            criterion.attrib = {'Name': f'C{j}', 'IsPreferenceIncreasing': '1'}
            criterion.append(comment)
            measurement = Et.Element('Measurement')
            measurement.attrib = {'Name': 'Ratings', 'Type': '0', 'DecisionMaker': 'Default User', 'SubType': '0'}
            measurement.append(comment)
            for k in range(start_alternatives, end_alternatives):
                ruler_value = Et.Element('RulerValue')
                try:
                    ruler_value.attrib = {'Name': f'A{k - 2}', 'Value': f'{p[j - 1][k - 3]}'}
                except IndexError:
                    ruler_value.attrib = {'Name': f'A{k - 2}', 'Value': f'{p[k - 3]}'}
                measurement.append(ruler_value)
            criterion.append(measurement)
            criteria_element.append(criterion)

        weights = Et.Element('Measurement')
        weights.attrib = {'Name': 'Criteria Weights', 'Type': '1', 'DecisionMaker': 'Default User', 'SubType': '0'}
        weights.append(comment)
        for n in range(1, criteria_count + 1):
            ruler_value = Et.Element('RulerValue')
            ruler_value.attrib = {'Name': f'C{n}', 'Value': f'{w[n - 1]}'}
            weights.append(ruler_value)
        criteria_element.append(weights)

        self._indent(root)
        tree.write(self.xml_file, encoding='utf-8', xml_declaration=True)
        return tree

    def request(self, w, p):
        """
        Method that handles the communication with the Tetra server. Gathers xml ElementTree from method make_xml and
        returns an array with the values that are returned from Tetra.

        :param w: Weights
        :param p: Population
        :return: Array
        """
        self.assertion_tests(w, p)
        if not p:
            return []

        xml_tree = Et.tostring(self._make_xml(w=w, p=p).getroot(), encoding='utf8', method='xml')

        try:
            r = post(self.url, data=xml_tree, auth=(self.user, self.password), headers=self.headers, verify=False,
                     timeout=10)
            r.raise_for_status()
        except HTTPError as err:
            raise HTTPError(f'HTTP Error occurred: {err}')

        values = []
        tree = Et.fromstring(r.content)
        self._indent(tree)
        Et.ElementTree(tree).write(f'{HERE}/tetra_out.xml', encoding='utf-8', xml_declaration=True)
        for items in tree.findall('Alternative'):
            values.append(-float(items.attrib.get('Value')))

        assert len(values) == len(p[0])
        return values

    @staticmethod
    def assertion_tests(w, p):
        """Assertion tests to check integrity of the data input"""
        assert len(w) == len(array(p)), 'Lists of weights and functions should be equal'
        assert round(sum(w), 4) == 1, f'Sum of weights should be equal to 1. They are now sum({w})={sum(w)}.'
        return
